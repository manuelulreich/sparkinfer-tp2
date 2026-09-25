// The tensor-parallel plan (dual-gpu WP-4): how many devices a model is split across,
// which devices, and the validation a load runs before it commits. Pure C++ -- no CUDA,
// no runtime linkage -- so the validation matrix is unit-testable on any box
// (runtime/tests/tp_config_cpu_test.cpp).
//
// This file is the single source of the gate's operator-facing messages: ModelEngine::load
// prints the strings below verbatim (each ends in \n), and the CPU test asserts them
// byte for byte, so rewording one here is a user-visible change.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace sparkinfer_server {

// The resolved TP plan: the tensor-parallel size, and the explicit device ids when the
// operator named any. An empty `devices` means "not given", in which case the effective
// list is the first `tp` ordinals -- at the default (tp=1, none given) that is {0}, i.e.
// today's single device 0, byte-identical.
struct TpPlan {
    int tp = 1;
    std::vector<int> devices;
};

// Process-scope plan state, set in sparkinfer_server.cpp::main from --tp/--devices and the
// SPARKINFER_TP/SPARKINFER_DEVICES env (flag wins over env, env over the defaults above).
// Defined in sparkinfer_server.cpp OUTSIDE its anonymous namespace, so this declaration and
// model_engine.cpp -- a separate translation unit -- all name the same two objects.
extern int g_tp;
extern std::vector<int> g_devices;

// The effective device list: the explicit ids if any were given, else the first `tp`
// ordinals. The tp=1 default is {0}.
inline std::vector<int> effective_devices(const TpPlan& plan) {
    if (!plan.devices.empty()) return plan.devices;
    std::vector<int> v;
    const int n = plan.tp > 0 ? plan.tp : 1;
    v.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) v.push_back(i);
    return v;
}

// The load-time gate (dual-gpu plan WP-4): enough devices for the requested tp, every
// requested id present. On failure `error` is the exact operator-facing refusal (ending
// in \n; the caller prints it as-is). The third check of the plan -- compute-capability
// mismatch across the requested cards -- cannot be pure C++; model_engine.cpp performs it
// after this one, with same_cc() below, and warns without failing.
struct TpValidation {
    bool ok = false;
    std::string error;  // non-empty when !ok
};

inline TpValidation validate_tp_plan(int tp, const std::vector<int>& requested, int ndev) {
    TpValidation r;
    if (ndev <= 0 && tp == 1) {
        // The legacy refusal, byte-identical to the pre-TP gate: tp=1 never gained a new
        // message here.
        r.error = "[sparkinfer-server] no CUDA device\n";
        return r;
    }
    if (ndev < tp) {
        r.error = "[sparkinfer-server] tensor parallelism: requested tp=" + std::to_string(tp) +
                  " but this box has " + std::to_string(ndev) + " CUDA device(s)\n";
        return r;
    }
    for (int id : requested) {
        if (id < 0 || id >= ndev) {
            r.error = "[sparkinfer-server] device " + std::to_string(id) +
                      " requested but not present (this box has devices 0.." +
                      std::to_string(ndev - 1) + ")\n";
            return r;
        }
    }
    r.ok = true;
    return r;
}

// Parses a `--devices a,b,...` value: digits-only ids (a leading '+' or '-' or any other
// non-digit rejects), at most 9 digits so the value cannot overflow int, optional spaces
// or tabs around each id, and no duplicate ids. On failure fills *err with the reason and
// returns false (the parsed list is emptied in all cases).
inline bool parse_device_list(const std::string& text, std::vector<int>* out, std::string* err) {
    out->clear();
    if (text.empty()) { *err = "empty device list"; return false; }
    if (text.back() == ',') { *err = "trailing comma in device list '" + text + "'"; return false; }

    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) end = text.size();
        std::string elem = text.substr(start, end - start);
        const size_t b = elem.find_first_not_of(" \t");
        if (b == std::string::npos) {
            *err = "empty element in device list '" + text + "'";
            return false;
        }
        const size_t e = elem.find_last_not_of(" \t");
        elem = elem.substr(b, e - b + 1);
        if (elem.size() > 9) {
            *err = "invalid device id '" + elem + "' in device list '" + text + "' (at most 9 digits)";
            return false;
        }
        for (char c : elem) {
            if (c < '0' || c > '9') {
                *err = "invalid device id '" + elem + "' in device list '" + text + "' (digits only)";
                return false;
            }
        }
        const int v = std::atoi(elem.c_str());
        if (std::find(out->begin(), out->end(), v) != out->end()) {
            *err = "duplicate device id " + elem + " in device list '" + text + "'";
            return false;
        }
        out->push_back(v);
        start = end + 1;
    }
    return true;
}

// Parses a `--tp N` value: digits only, at most 9 (so it cannot overflow int), and >= 1
// (tp 0 is not a plan).
inline bool parse_tp_value(const std::string& text, int* out) {
    if (text.empty() || text.size() > 9) return false;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
    }
    const int v = std::atoi(text.c_str());
    if (v < 1) return false;
    *out = v;
    return true;
}

// Compute-capability equality for the warn-only arch check: the split is only safe across
// cards of the same major.minor.
inline bool same_cc(int a_major, int a_minor, int b_major, int b_minor) {
    return a_major == b_major && a_minor == b_minor;
}

}  // namespace sparkinfer_server
