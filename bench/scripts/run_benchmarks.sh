#!/bin/bash
# Decode + prefill ladder against the sparkinfer server already running on this box (tp=1 or
# tp=2). Prints one ROW line per measurement; see simple_bench.py for the method.
#
#   bash bench/scripts/run_benchmarks.sh                              # up to 16k (the default)
#   bash bench/scripts/run_benchmarks.sh --max-context 65536
#   bash bench/scripts/run_benchmarks.sh --conc 1,2,4 --decode-prompt count
#
# HOST/PORT select the server (default 127.0.0.1:8080), OPENAI_API_KEY its key if one is bound,
# OUT the raw-response file. Every other argument goes to simple_bench.py (--help lists them).
# Start the server first, e.g. for two cards:
#   ./build/server/sparkinfer_server -m $M --tokenizer $M/tokenizer.json --tp 2 --devices 0,1 \
#       --ctx 32768 [--draft-model $D]
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$HERE/../results/bench_$(date +%Y%m%d_%H%M%S).jsonl}
mkdir -p "$(dirname "$OUT")"
exec python3 "$HERE/simple_bench.py" --out "$OUT" "$@"
