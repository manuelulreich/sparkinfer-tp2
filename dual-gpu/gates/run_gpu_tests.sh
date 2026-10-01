#!/bin/bash
# WP-15 (1): run every single-GPU test once per card, then the 2-GPU and tp CPU tests.
#
# The single-GPU tests take no device argument; CUDA_VISIBLE_DEVICES pins each run to one card,
# so the suite needs no code change. Exit status is the number of failing runs (0 = all green).
#
#   dual-gpu/gates/run_gpu_tests.sh [build_dir]     # default: build
set -u
BUILD="${1:-build}"
T="$BUILD/runtime/tests"
[ -d "$T" ] || { echo "no test directory at $T (build first)" >&2; exit 2; }
ndev=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$ndev" -ge 1 ] || { echo "no GPU visible" >&2; exit 2; }

fails=0
run() {   # run <label> <cmd...>
    local label="$1"; shift
    local out rc
    out=$("$@" 2>&1); rc=$?
    local note=""
    echo "$out" | grep -qi "skip" && note=" (skipped: see the test's own output)"
    if [ $rc -eq 0 ]; then echo "PASS  $label$note"; else echo "FAIL  $label (rc=$rc)"; fails=$((fails + 1)); fi
}

for d in $(seq 0 $((ndev - 1))); do
    for t in $(ls "$T" | grep '_gpu_test$' | sort); do
        run "dev$d $t" env CUDA_VISIBLE_DEVICES=$d timeout 600 "$T/$t"
    done
done
if [ "$ndev" -ge 2 ]; then
    # Both link transports (p2p-mapped and pinned-staging), the flag all-reduce, max-reduce,
    # the cross-context canary and the churn gate.
    run "gpu_link_2gpu_test" timeout 600 "$T/gpu_link_2gpu_test"
fi
for t in gdn_split_cpu_test tp_weights_cpu_test tp_config_cpu_test tp_layout_cpu_test \
         kv_split_cpu_test device_health_cpu_test gpu_metrics_cpu_test scheduler_cpu_test; do
    [ -x "$T/$t" ] && run "$t" timeout 600 "$T/$t"
done
echo "failures: $fails"
exit $fails
