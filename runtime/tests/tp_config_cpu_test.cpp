// The TP gate's validation matrix (dual-gpu WP-4): pure C++ against
// server/include/tp_plan.hpp -- no CUDA, no GPU, no runtime linkage, so it pins the
// operator-facing refusals byte for byte on any box:
//   * the legacy "no CUDA device" refusal, unchanged for tp=1
//   * the tp > device-count refusal
//   * the requested-id-out-of-range refusal
//   * --tp / --devices value parsing (digits only, range, empty, commas, duplicates)
//   * the architecture-match helper the warn-only cross-card check builds on
//
// Build: g++ -std=c++17 tp_config_cpu_test.cpp -o tp_config_cpu_test

#include "../../server/include/tp_plan.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using sparkinfer_server::TpPlan;
using sparkinfer_server::TpValidation;
using sparkinfer_server::effective_devices;
using sparkinfer_server::parse_device_list;
using sparkinfer_server::parse_tp_value;
using sparkinfer_server::same_cc;
using sparkinfer_server::validate_tp_plan;

void test_default_plan_shape() {
    // The flag/env-absent default: tp=1, no explicit devices -- the single device 0,
    // i.e. today's behaviour.
    const TpPlan def;
    CHECK(def.tp == 1);
    CHECK(def.devices.empty());
    const std::vector<int> eff = effective_devices(def);
    CHECK(eff.size() == 1 && eff[0] == 0);

    // tp=2 without explicit devices resolves to the first two ordinals.
    const TpPlan tp2{2, {}};
    const std::vector<int> e2 = effective_devices(tp2);
    CHECK(e2.size() == 2 && e2[0] == 0 && e2[1] == 1);
}

void test_tp_larger_than_the_box() {
    // tp=2 on a one-GPU box: the exact refusal, trailing newline included.
    TpValidation v = validate_tp_plan(2, effective_devices(TpPlan{2, {}}), 1);
    CHECK(!v.ok);
    CHECK(v.error == "[sparkinfer-server] tensor parallelism: requested tp=2 but this box has "
                     "1 CUDA device(s)\n");
    // Same refusal on a box with no CUDA at all (tp>1 must not emit the legacy message).
    v = validate_tp_plan(2, effective_devices(TpPlan{2, {}}), 0);
    CHECK(!v.ok);
    CHECK(v.error == "[sparkinfer-server] tensor parallelism: requested tp=2 but this box has "
                     "0 CUDA device(s)\n");
}

void test_legacy_no_device_message_unchanged() {
    // tp=1 with no devices and no CUDA: byte-identical to the pre-TP gate.
    TpValidation v = validate_tp_plan(1, effective_devices(TpPlan{1, {}}), 0);
    CHECK(!v.ok);
    CHECK(v.error == "[sparkinfer-server] no CUDA device\n");
}

void test_requested_id_must_exist() {
    // tp=1 with an explicit id past the last card: exact refusal, range stated.
    TpValidation v = validate_tp_plan(1, {5}, 2);
    CHECK(!v.ok);
    CHECK(v.error == "[sparkinfer-server] device 5 requested but not present "
                     "(this box has devices 0..1)\n");
    // The tp=2 explicit pair on a two-card box: valid.
    v = validate_tp_plan(2, {0, 1}, 2);
    CHECK(v.ok);
    // tp=2 with one explicit card on a two-card box: the gate (count >= tp, ids present)
    // passes; mapping ranks to the remaining card is the layout layer's concern, not
    // this gate's.
    v = validate_tp_plan(2, {0}, 2);
    CHECK(v.ok);
}

void test_parse_tp_value() {
    int n = -1;
    CHECK(!parse_tp_value("0", &n));          // below 1
    CHECK(!parse_tp_value("00", &n));         // still below 1
    CHECK(!parse_tp_value("", &n));
    CHECK(!parse_tp_value("1x", &n));
    CHECK(!parse_tp_value("x1", &n));
    CHECK(!parse_tp_value("-1", &n));
    CHECK(!parse_tp_value("+1", &n));
    CHECK(!parse_tp_value(" 1", &n));
    CHECK(!parse_tp_value("1 1", &n));
    CHECK(!parse_tp_value("9999999999", &n));  // ten digits: would overflow int
    CHECK(parse_tp_value("1", &n) && n == 1);
    CHECK(parse_tp_value("2", &n) && n == 2);
    CHECK(parse_tp_value("12", &n) && n == 12);
    CHECK(parse_tp_value("999999999", &n) && n == 999999999);
}

void test_parse_device_list() {
    std::vector<int> v;
    std::string err;

    // The plain cases: comma-separated, optional spaces/tabs around each id.
    CHECK(parse_device_list("0,1", &v, &err));
    CHECK(v.size() == 2 && v[0] == 0 && v[1] == 1);
    CHECK(parse_device_list("1, 2", &v, &err));
    CHECK(v.size() == 2 && v[0] == 1 && v[1] == 2);
    CHECK(parse_device_list("3", &v, &err));
    CHECK(v.size() == 1 && v[0] == 3);
    CHECK(parse_device_list(" \t4\t ", &v, &err));
    CHECK(v.size() == 1 && v[0] == 4);

    // Duplicates are refused: the operator meant one card twice, or it is a typo.
    CHECK(!parse_device_list("1,1", &v, &err));
    CHECK(!err.empty());
    CHECK(!parse_device_list("0,1,0", &v, &err));
    CHECK(!err.empty());

    // No signs, no non-digit characters, no empty elements, no trailing comma.
    CHECK(!parse_device_list("-1", &v, &err));
    CHECK(!parse_device_list("1 2", &v, &err));
    CHECK(!parse_device_list("a", &v, &err));
    CHECK(!parse_device_list("1,", &v, &err));
    CHECK(!parse_device_list(",1", &v, &err));
    CHECK(!parse_device_list("1,,2", &v, &err));
    CHECK(!parse_device_list("", &v, &err));
    CHECK(!parse_device_list("9999999999", &v, &err));  // ten digits
}

void test_same_cc() {
    CHECK(same_cc(12, 0, 12, 0));
    CHECK(same_cc(8, 9, 8, 9));
    CHECK(!same_cc(12, 0, 12, 1));  // same major, different minor: different arch
    CHECK(!same_cc(9, 0, 12, 0));
}

int main() {
    test_default_plan_shape();
    test_tp_larger_than_the_box();
    test_legacy_no_device_message_unchanged();
    test_requested_id_must_exist();
    test_parse_tp_value();
    test_parse_device_list();
    test_same_cc();
    std::printf("tp_config_cpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
