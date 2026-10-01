#!/usr/bin/env bash
# Smoke test for a two-card box: start the release image split across both GPUs (TP=2), wait for
# health, check /v1/info reports tp=2 with two devices, and run one greedy completion.
#
#   docker/smoke-tp2.sh [image] [serve-dspark]
#
# Uses the same named volume as the README (weights download on first run). Exit 0 = pass.
set -euo pipefail
IMAGE="${1:-ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:latest}"
MODE="${2:-}"
PORT="${PORT:-8080}"
NAME="sparkinfer-smoke-tp2-$$"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" --gpus all -p "$PORT:8080" -v qwen38:/models \
  -e TP=2 -e DEVICES="${DEVICES:-0,1}" "$IMAGE" ${MODE:+"$MODE"} >/dev/null

echo "[smoke-tp2] waiting for the server (first run downloads the weights)..."
for _ in $(seq 1 1800); do
  if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! docker ps -q --filter "name=$NAME" | grep -q .; then
    echo "[smoke-tp2] FAIL: container exited during startup" >&2
    docker logs "$NAME" 2>&1 | tail -30 >&2
    exit 1
  fi
  sleep 2
done
curl -fs "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[smoke-tp2] FAIL: never healthy" >&2; exit 1; }

info=$(curl -fs "http://127.0.0.1:$PORT/v1/info")
python3 - "$info" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert d.get("tp") == 2, f"tp is {d.get('tp')}, expected 2"
assert len(d.get("devices") or []) == 2, "expected two devices"
assert d.get("healthy") is True, "server reports unhealthy"
print(f"[smoke-tp2] /v1/info: tp=2, link={d.get('link')}, devices="
      + ", ".join(f"{x.get('device')}:{x.get('name')}" for x in d["devices"]))
PY

out=$(curl -fs "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0}')
python3 - "$out" <<'PY'
import json, sys
text = json.loads(sys.argv[1])["choices"][0]["text"]
assert "Paris" in text, f"unexpected completion: {text!r}"
print(f"[smoke-tp2] completion ok: {text!r}")
PY
echo "[smoke-tp2] PASS"
