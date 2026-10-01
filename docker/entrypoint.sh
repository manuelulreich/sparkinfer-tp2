#!/usr/bin/env bash
# Default: serve (AR + image + video) over an OpenAI-compatible API.
# `serve-dspark`: download the drafter and enable DSpark for eligible requests.
# `bench [workload]`: run the DSpark speculative benchmark instead.
set -euo pipefail

fetch() {  # repo dir label
  if [ ! -f "$2/config.json" ]; then
    # SPARKINFER_NO_DOWNLOAD=1: the weights are pre-staged and the container has no route out
    # (#1090). Say what is missing instead of failing inside a download that cannot work.
    if [ "${SPARKINFER_NO_DOWNLOAD:-0}" = "1" ]; then
      echo "[sparkinfer] SPARKINFER_NO_DOWNLOAD=1 and $2 holds no config.json: mount the $3 weights there, or point the matching *_DIR at them." >&2
      exit 1
    fi
    echo "[sparkinfer] downloading $3 ($1) — first run only, cached in /models"
    hf download "$1" --local-dir "$2"
  fi
}

if [ "${1:-}" = "bench" ]; then
  shift
  WORKLOAD="${1:-code}"
  fetch "$MODEL_REPO" "$MODEL_DIR" "target"
  fetch "$DRAFT_REPO" "$DRAFT_DIR" "DSpark drafter"
  echo "[sparkinfer] DSpark bench · workload=$WORKLOAD · ${BENCH_TOKENS:-256} tokens"
  MODEL_DIR="$MODEL_DIR" python3 /opt/sparkinfer/mkids.py "$WORKLOAD" > /tmp/ids.txt
  exec /opt/sparkinfer/bin/qwen38_hf_dflash_bench \
       "$MODEL_DIR" "$DRAFT_DIR" "${BENCH_TOKENS:-256}" $(cat /tmp/ids.txt)
fi

# Two cards (tensor parallel): -e TP=2 [-e DEVICES=0,1] splits the model across both. Measured on
# 2x 16 GB (RTX 5060 Ti): 131,072 tokens fit without the draft; with DSpark the draft's card leaves
# room for 49,152 (the server refuses a larger --ctx at load rather than failing per request).
TP_ARGS=()
if [ -n "${TP:-}" ] && [ "${TP}" != "1" ]; then
  TP_ARGS+=(--tp "$TP")
  [ -n "${DEVICES:-}" ] && TP_ARGS+=(--devices "$DEVICES")
fi

if [ "${1:-}" = "serve-dspark" ]; then
  shift
  fetch "$DRAFT_REPO" "$DRAFT_DIR" "DSpark drafter"
  export SPARKINFER_DRAFT_MODEL="$DRAFT_DIR"
  # The drafter needs device memory the full 262,144-token KV pool leaves no room for on a 32 GB
  # card (#1086), so DSpark defaults to half the context. -e CTX=... (or --ctx) still overrides.
  if [ "${#TP_ARGS[@]}" -gt 0 ]; then CTX="${CTX:-49152}"; else CTX="${CTX:-131072}"; fi
fi
# Autoregressive serving fits the model's full context on one 32 GB card; two 16 GB cards hold half.
if [ "${#TP_ARGS[@]}" -gt 0 ]; then CTX="${CTX:-131072}"; else CTX="${CTX:-262144}"; fi

fetch "$MODEL_REPO" "$MODEL_DIR" "target"
MODE="autoregressive"
[ -n "${SPARKINFER_DRAFT_MODEL:-}" ] && MODE="DSpark ($SPARKINFER_DRAFT_MODEL)"
echo "[sparkinfer] serving $MODEL_DIR as '$MODEL_NAME' on $HOST:$PORT (ctx $CTX, max output $SPARKINFER_MAX_OUTPUT_TOKENS, $MODE${TP:+, tp $TP})"
exec /opt/sparkinfer/bin/sparkinfer_server \
  -m "$MODEL_DIR" --tokenizer "$MODEL_DIR/tokenizer.json" \
  --model-name "$MODEL_NAME" --ctx "$CTX" --host "$HOST" --port "$PORT" ${TP_ARGS[@]+"${TP_ARGS[@]}"} "$@"
