#!/usr/bin/env bash
# Shared helpers for the sparkinfer bench / accuracy scripts.
# Sourced by bench.sh and accuracy.sh. Everything auto-detects / auto-builds so a
# contributor can run a single command on a fresh Blackwell box.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (bench/scripts -> root)
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-30B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-30B-A3B-Q4_K_M.gguf}"
TOK_REPO="${TOK_REPO:-Qwen/Qwen3-30B-A3B}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$ROOT/.llamacpp}"   # override to reuse an existing checkout

# C2 (reference quarantine): pin the baseline artifacts so a tampered persisted copy can't skew a
# verdict. reference.lock carries MODEL_SHA256 + LLAMACPP_COMMIT; empty = warn-only until pinned.
#
# Captured BEFORE the lock is sourced: reference.lock gives MODEL_SHA256 a `:-` default, so after
# sourcing it is always non-empty and an explicit caller pin becomes indistinguishable from the
# default. resolve_model_sha256() below needs that distinction.
_ENV_MODEL_SHA256="${MODEL_SHA256:-}"
_HERE_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_HERE_COMMON/reference.lock" ] && source "$_HERE_COMMON/reference.lock"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
sha256_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }

# compute capability -> CUDA arch (12.0 -> 120). RTX 5090 / PRO 6000 = 120, Spark/Thor = 121.
detect_arch() {
  local cc arch="${ARCH:-}"
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:].')"
  if [[ ! "$cc" =~ ^[0-9]+$ ]]; then cc=120; fi
  if [[ -n "$arch" && "$arch" =~ ^[0-9]+$ ]]; then
    echo "$arch"
  else
    echo "$cc"
  fi
}

# Bare-metal SSH boxes often install nvcc outside default PATH (non-interactive ssh).
for _cuda in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_cuda/bin/nvcc" ] && export PATH="$_cuda/bin:$PATH" && export CUDA_HOME="$_cuda" && break
done
unset _cuda

# nvcc 12.8 fails against Ubuntu 24.04's GCC 13.3 libstdc++ (cstdio / __gnu_cxx errors). Pin the
# CUDA host compiler to g++-12 (a fully supported combo) when it's available.
CUDA_HOST_FLAG=""
[ -x /usr/bin/g++-12 ] && CUDA_HOST_FLAG="-DCMAKE_CUDA_HOST_COMPILER=g++-12"

ensure_sparkinfer() {  # $1 = arch
  [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ] && [ -x "$ROOT/build/runtime/qwen3_gguf_score" ] \
    && [ -x "$ROOT/build/runtime/qwen3_gguf_prefill_check" ] && return 0
  echo ">> building sparkinfer (sm_$1) ..." >&2
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_CUDA_ARCHITECTURES="$1" -DCMAKE_BUILD_TYPE=Release $CUDA_HOST_FLAG >/dev/null
  # Cap at 2 parallel jobs — cc1plus for sm_120 uses ~2-3 GB RAM each; -j4 OOMs on 64GB eval boxes.
  if ! cmake --build "$ROOT/build" -j2 >/dev/null; then
    echo ">> sparkinfer build FAILED (sm_$1)" >&2
    return 1
  fi
  [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ] && [ -x "$ROOT/build/runtime/qwen3_gguf_score" ] \
    && [ -x "$ROOT/build/runtime/qwen3_gguf_prefill_check" ] || {
    echo ">> sparkinfer build incomplete — missing runtime binaries (sm_$1)" >&2
    return 1
  }
}

# H3: batched-vs-token prefill fidelity binary (not always in release tarballs).
ensure_prefill_check() {  # $1 = arch
  if [ -x "$ROOT/build/runtime/qwen3_gguf_prefill_check" ]; then return 0; fi
  if [ -n "${SI_BIN:-}" ] && [ -x "$SI_BIN/qwen3_gguf_prefill_check" ]; then return 0; fi
  echo ">> building qwen3_gguf_prefill_check (sm_$1) ..." >&2
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_CUDA_ARCHITECTURES="$1" -DCMAKE_BUILD_TYPE=Release $CUDA_HOST_FLAG >/dev/null
  cmake --build "$ROOT/build" -j2 --target qwen3_gguf_prefill_check >/dev/null
  [ -x "$ROOT/build/runtime/qwen3_gguf_prefill_check" ]
}

# Mixed-load CB TTFT bench (prefill scoring when SPARKINFER_EVAL_PREFILL_CB=1).
ensure_cb_bench() {  # $1 = arch
  if [ -x "$ROOT/build/runtime/qwen3_gguf_cb_bench" ]; then return 0; fi
  if [ -n "${SI_BIN:-}" ] && [ -x "$SI_BIN/qwen3_gguf_cb_bench" ]; then return 0; fi
  echo ">> building qwen3_gguf_cb_bench (sm_$1) ..." >&2
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_CUDA_ARCHITECTURES="$1" -DCMAKE_BUILD_TYPE=Release $CUDA_HOST_FLAG >/dev/null
  cmake --build "$ROOT/build" -j2 --target qwen3_gguf_cb_bench >/dev/null
  [ -x "$ROOT/build/runtime/qwen3_gguf_cb_bench" ]
}

_download_model() {
  echo ">> downloading $MODEL_REPO/$MODEL_FILE -> $MODELS_DIR (~17 GB) ..." >&2
  mkdir -p "$MODELS_DIR"
  # Try three download methods in order; fall back to plain curl (no HF tools needed).
  HF_HUB_DISABLE_XET=1 hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR" >&2 || \
  python3 -c "from huggingface_hub import hf_hub_download as d; d('$MODEL_REPO','$MODEL_FILE',local_dir='$MODELS_DIR')" >&2 || \
  curl -fL --progress-bar "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
       -o "$MODELS_DIR/$MODEL_FILE" >&2
}

ensure_model() {
  [ -f "$MODELS_DIR/$MODEL_FILE" ] || _download_model
  verify_model
}

# Point MODEL_SHA256 at the digest that belongs to the CURRENT $MODEL_FILE.
#
# reference.lock pins one sha per model, but MODEL_SHA256's own default is the Qwen3-30B guard
# model's -- it is the default because _common.sh's own default MODEL_FILE is that model. Callers
# that serve something else must therefore pin the matching digest, which every eval path does by
# hand (evaluate_dual.sh, evaluate_triple.sh pass MODEL_SHA256="$QWEN36_MODEL_SHA256"). server/run.sh
# did not, so serving Qwen3.6 verified 22 GB of correct weights against the 30B guard's digest,
# deleted them, re-downloaded, and failed again -- an infinite re-fetch loop under a restarting
# supervisor (reported by the Gittensor serving integration, 2026-08-23; the HF upload was never
# stale). Resolving from MODEL_FILE makes that class of mismatch structurally impossible.
#
# An explicit MODEL_SHA256 in the ENVIRONMENT always wins: the caller pinning the digest it expects
# is the whole point of the knob, and downstream consumers (the subnet passes its own pin) rely on
# it. An unrecognised MODEL_FILE resolves to empty => warn-only, which is correct: verifying a file
# against some other file's digest is exactly the bug this function exists to prevent.
resolve_model_sha256() {
  if [ -n "${_ENV_MODEL_SHA256:-}" ]; then
    MODEL_SHA256="$_ENV_MODEL_SHA256"
    return 0
  fi
  case "$MODEL_FILE" in
    Qwen3.6-35B-A3B-UD-Q4_K_M.gguf)             MODEL_SHA256="${QWEN36_MODEL_SHA256:-}" ;;
    Qwen3-30B-A3B-Q4_K_M.gguf)                  MODEL_SHA256="${MODEL_SHA256:-}" ;;
    Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf)  MODEL_SHA256="${QWEN35_9B_Q4K_SHA256:-}" ;;
    Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf)    MODEL_SHA256="${QWEN35_9B_Q8_SHA256:-}" ;;
    Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf)    MODEL_SHA256="${QWEN35_9B_BF16_SHA256:-}" ;;
    *)                                          MODEL_SHA256="" ;;
  esac
  export MODEL_SHA256
}

# C2: the persisted baseline weights must be pristine — a malicious root build could corrupt the GGUF
# to depress llama's score and inflate its own relative gain. Verify against the pinned sha each eval.
#
# MODEL_SHA_POLICY selects what a mismatch means:
#   refetch (default) — the eval's C2 quarantine: assume the PIN is truth and the local copy is
#                       suspect, so delete and re-download once, then fail. Right for the eval,
#                       where a tampered persisted baseline is the threat being defended against.
#   strict            — assume the FILE is truth and report the mismatch. Right for serving: a
#                       serving start has no adversary to quarantine against, and deleting 22 GB
#                       on every start is destructive when it is the pin that is wrong. Under a
#                       restarting supervisor, refetch turns a bad pin into an unbounded download
#                       loop; strict fails once, loudly, with both digests.
verify_model() {
  local f="$MODELS_DIR/$MODEL_FILE" got policy="${MODEL_SHA_POLICY:-refetch}"
  got="$(sha256_of "$f")"
  if [ -z "${MODEL_SHA256:-}" ]; then
    echo ">> model sha256 (not pinned, warn-only): $got" >&2; return 0
  fi
  if [ "$got" = "$MODEL_SHA256" ]; then echo ">> model sha256 OK" >&2; return 0; fi
  if [ "$policy" = "strict" ]; then
    echo ">> FATAL: model sha256 MISMATCH for $MODEL_FILE" >&2
    echo ">>   got  ${got:-<file missing or unreadable>}" >&2
    echo ">>   want $MODEL_SHA256" >&2
    echo ">> Refusing to delete and re-download. Either the file is not the pinned build, or the" >&2
    echo ">> pin is wrong for this file. Pass MODEL_SHA256=<digest> to pin the copy you intend to" >&2
    echo ">> serve, or MODEL_SHA_POLICY=refetch to re-fetch a clean baseline." >&2
    return 1
  fi
  echo ">> WARN: model sha256 MISMATCH (got ${got:-none}, want $MODEL_SHA256) — re-fetching clean baseline" >&2
  rm -f "$f"; _download_model
  got="$(sha256_of "$f")"
  [ "$got" = "$MODEL_SHA256" ] || { echo ">> FATAL: model sha still wrong after re-download ($got)" >&2; return 1; }
  echo ">> model sha256 OK after re-fetch" >&2
}

# Download tokenizer.json for TOK_REPO into MODELS_DIR. Refuses to reuse a leftover from a
# different repo — a Qwen3-30B tokenizer (vocab ~151k) in models35 silently tanks Qwythos
# teacher-forced accuracy (~0.88 top1 / KL~0.20) while long-context still looks fine.
ensure_tokenizer() {
  local marker="$MODELS_DIR/.tokenizer_repo" need=0 vocab=0
  mkdir -p "$MODELS_DIR"
  [ -f "$MODELS_DIR/tokenizer.json" ] || need=1
  [ -f "$marker" ] && [ "$(cat "$marker" 2>/dev/null)" = "$TOK_REPO" ] || need=1
  if [ "$need" = 0 ] && command -v python3 >/dev/null; then
    vocab="$(python3 -c "from tokenizers import Tokenizer; print(Tokenizer.from_file(r'$MODELS_DIR/tokenizer.json').get_vocab_size())" 2>/dev/null || echo 0)"
    case "$TOK_REPO" in
      *Qwen3.5*|*Qwen3.6*|*Qwen3_5*|*Qwen3_6*)
        # Qwythos / Qwen3.5 / Qwen3.6 GGUFs embed ~248k tokens.
        [ "${vocab:-0}" -ge 200000 ] || need=1 ;;
      *Qwen3-30B*|*Qwen3_30B*)
        [ "${vocab:-0}" -ge 140000 ] && [ "${vocab:-0}" -lt 200000 ] || need=1 ;;
    esac
  fi
  if [ "$need" = 0 ]; then return 0; fi
  echo ">> downloading tokenizer.json from $TOK_REPO (replacing mismatched/missing) ..." >&2
  # Download into a staging dir and only move into place once the file exists. The previous order
  # -- rm the existing tokenizer FIRST, then fetch -- turns any transient network failure into a
  # models dir with no tokenizer at all, which is strictly worse than the possibly-stale one it
  # replaced, and leaves the server unable to start until someone re-fetches by hand. Observed for
  # real on 2026-08-23: one `curl: (56) Failure when receiving data from the peer` deleted a
  # perfectly good tokenizer and bricked an otherwise healthy box.
  local stage="$MODELS_DIR/.tokenizer_stage"
  rm -rf "$stage"; mkdir -p "$stage"
  HF_HUB_DISABLE_XET=1 hf download "$TOK_REPO" tokenizer.json --local-dir "$stage" >&2 || \
  python3 -c "from huggingface_hub import hf_hub_download as d; d('$TOK_REPO','tokenizer.json',local_dir='$stage')" >&2 || \
  curl -fL --retry 3 --retry-all-errors --progress-bar "https://huggingface.co/${TOK_REPO}/resolve/main/tokenizer.json" \
       -o "$stage/tokenizer.json" >&2
  if [ ! -s "$stage/tokenizer.json" ]; then
    rm -rf "$stage"
    echo "!! tokenizer download failed for $TOK_REPO" >&2
    if [ -f "$MODELS_DIR/tokenizer.json" ]; then
      echo "!! keeping the existing tokenizer.json -- it may be the wrong one for $TOK_REPO." >&2
      echo "!! Delete it and re-run once the network is back if scores look wrong." >&2
    fi
    return 1
  fi
  mv -f "$stage/tokenizer.json" "$MODELS_DIR/tokenizer.json"
  rm -rf "$stage"
  printf '%s\n' "$TOK_REPO" > "$marker"
}

# Reuse the persisted llama.cpp only if it's the pinned commit, a clean tree, and the binary still
# matches the hash recorded when it was built (a root PR build can't have swapped it). Else rebuild.
_llamacpp_clean() {  # $1=llama-bench  $2=sentinel
  if [ -n "${LLAMACPP_COMMIT:-}" ]; then
    [ "$(git -C "$LLAMACPP_DIR" rev-parse HEAD 2>/dev/null)" = "$(git -C "$LLAMACPP_DIR" rev-parse "$LLAMACPP_COMMIT^{commit}" 2>/dev/null)" ] || return 1
    [ -z "$(git -C "$LLAMACPP_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ] || return 1
  fi
  [ -f "$2" ] && [ "$(sha256_of "$1")" = "$(cat "$2" 2>/dev/null)" ]
}

_llamacpp_binary_ok() {  # headless llama-bench is multi-MB; partial links are ~17KB
  local f="$1" sz
  [ -x "$f" ] || return 1
  sz="$(stat -c%s "$f" 2>/dev/null || wc -c <"$f" 2>/dev/null || echo 0)"
  [ "${sz:-0}" -gt 500000 ]
}

_llamacpp_purge_stale_build() {  # $1=bdir — drop UI tree when headless server build is required
  local bdir="$1"
  if [ -d "$bdir/tools/ui" ]; then
    echo ">> llama.cpp purge stale UI build tree (headless server) ..." >&2
    rm -rf "$bdir"
    return 0
  fi
  if [ -f "$bdir/CMakeCache.txt" ]; then
    if ! grep -q 'LLAMA_BUILD_UI:BOOL=OFF' "$bdir/CMakeCache.txt" 2>/dev/null || \
       ! grep -q 'LLAMA_USE_PREBUILT_UI:BOOL=OFF' "$bdir/CMakeCache.txt" 2>/dev/null; then
      echo ">> llama.cpp reconfigure (headless server, no prebuilt UI) ..." >&2
      rm -rf "$bdir"
    fi
  fi
}

ensure_llamacpp() {  # $1 = arch ; builds llama-bench + llama-server, pinned + tamper-checked (C2)
  local bench="$LLAMACPP_DIR/build/bin/llama-bench" srv="$LLAMACPP_DIR/build/bin/llama-server"
  local sentinel="$LLAMACPP_DIR/.si_refhash"
  local arch="$1" bdir="$LLAMACPP_DIR/build"
  # Stale partial trees (UI assets, old CMakeCache) leave llama-server broken while llama-bench exists.
  if [ ! -x "$srv" ] && { [ -d "$bdir/tools/ui" ] || [ -f "$bdir/CMakeCache.txt" ]; }; then
    echo ">> llama.cpp llama-server missing — purging stale build tree ..." >&2
    rm -rf "$bdir"
    rm -f "$sentinel"
  fi
  [ -x "$bench" ] && [ -x "$srv" ] && _llamacpp_clean "$bench" "$sentinel" && return 0
  echo ">> (re)building llama.cpp reference (CUDA sm_$arch) ..." >&2
  if [ -n "${LLAMACPP_COMMIT:-}" ]; then
    local head_ok=0
    if [ -d "$LLAMACPP_DIR/.git" ]; then
      [ "$(git -C "$LLAMACPP_DIR" rev-parse HEAD 2>/dev/null)" = \
        "$(git -C "$LLAMACPP_DIR" rev-parse "$LLAMACPP_COMMIT^{commit}" 2>/dev/null)" ] && head_ok=1
    fi
    if [ "$head_ok" != 1 ]; then
      rm -rf "$LLAMACPP_DIR"; mkdir -p "$LLAMACPP_DIR"
      git -C "$LLAMACPP_DIR" init -q
      git -C "$LLAMACPP_DIR" remote add origin "$LLAMACPP_REPO"
      git -C "$LLAMACPP_DIR" fetch -q --depth 1 origin "$LLAMACPP_COMMIT" >&2 || {
        echo ">> FATAL: cannot fetch pinned llama commit $LLAMACPP_COMMIT" >&2; return 1; }
      git -C "$LLAMACPP_DIR" checkout -q FETCH_HEAD
      rm -rf "$bdir"
    fi
  else
    [ -d "$LLAMACPP_DIR/.git" ] || git clone --depth=1 "$LLAMACPP_REPO" "$LLAMACPP_DIR" >&2
    echo ">> llama.cpp NOT pinned (warn-only) — HEAD $(git -C "$LLAMACPP_DIR" rev-parse --short HEAD 2>/dev/null); set LLAMACPP_COMMIT in reference.lock" >&2
    rm -rf "$bdir"
  fi
  _llamacpp_purge_stale_build "$bdir"
  if [ ! -f "$bdir/CMakeCache.txt" ]; then
    : > /tmp/llama_build.log
    if ! cmake -S "$LLAMACPP_DIR" -B "$bdir" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" \
          -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON \
          -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
          $CUDA_HOST_FLAG >&2; then
      echo ">> FATAL: llama.cpp cmake configure failed" >&2; return 1
    fi
  fi
  # Build targets sequentially — parallel llama-bench + llama-server races on shared ggml objects.
  if [ ! -x "$bench" ]; then
    echo ">> building llama-bench ..." >&2
    if ! cmake --build "$bdir" -j2 --target llama-bench 2>&1 \
          | tee -a /tmp/llama_build.log | awk 'NR<=5 || NR%40==0 || /Built target|error:|FAILED|fatal error/' >&2; then
      echo ">> FATAL: llama-bench build failed" >&2
      grep -iE 'error:|fatal error:' /tmp/llama_build.log | tail -20 >&2 || tail -40 /tmp/llama_build.log >&2
      return 1
    fi
  fi
  if [ ! -x "$srv" ]; then
    echo ">> building llama-server (single-threaded) ..." >&2
    if ! cmake --build "$bdir" -j1 --target llama-server 2>&1 \
          | tee -a /tmp/llama_build.log | awk 'NR<=5 || NR%20==0 || /Built target|error:|FAILED|fatal error/' >&2; then
      echo ">> FATAL: llama-server build failed" >&2
      grep -iE 'error:|fatal error:' /tmp/llama_build.log | tail -20 >&2 || tail -40 /tmp/llama_build.log >&2
      return 1
    fi
  fi
  [ -x "$bench" ] && [ -x "$srv" ] || { echo ">> FATAL: llama binaries missing after build" >&2; return 1; }
  sha256_of "$bench" > "$sentinel" 2>/dev/null || true
  echo ">> llama.cpp reference ready ($(git -C "$LLAMACPP_DIR" rev-parse --short HEAD 2>/dev/null))" >&2
}

# ---- prebuilt binaries (GitHub release) with source-build fallback ----
PREBUILT_REPO="${PREBUILT_REPO:-gittensor-ai-lab/sparkinfer}"
PREBUILT_TAG="${PREBUILT_TAG:-latest}"  # latest = newest release that carries a matching binary asset
PREBUILT_TGZ="${PREBUILT_TGZ:-}"
PREBUILT_URL="${PREBUILT_URL:-}"
PREBUILT_DIR=""
SI_BIN=""; SI_LD=""   # set by resolve_runner: binary dir + LD_LIBRARY_PATH

resolve_prebuilt_release() {   # $1=arch. Sets PREBUILT_TAG/PREBUILT_TGZ/PREBUILT_URL/PREBUILT_DIR.
  local arch="$1" tag="$PREBUILT_TAG" asset="" url="" cache_tag=""

  if [ -n "$PREBUILT_URL" ]; then
    asset="${PREBUILT_TGZ:-$(basename "${PREBUILT_URL%%\?*}")}"
    cache_tag="${tag:-custom}"
  elif [ -n "$PREBUILT_TGZ" ]; then
    asset="$PREBUILT_TGZ"
    if [ "$tag" = "latest" ]; then
      if [[ "$asset" =~ ^sparkinfer-(v[0-9][^-]*)- ]]; then
        tag="${BASH_REMATCH[1]}"
        url="https://github.com/$PREBUILT_REPO/releases/download/$tag/$asset"
      else
        url="https://github.com/$PREBUILT_REPO/releases/latest/download/$asset"
      fi
    else
      url="https://github.com/$PREBUILT_REPO/releases/download/$tag/$asset"
    fi
    cache_tag="$tag"
  elif [ "$tag" = "latest" ]; then
    command -v python3 >/dev/null || return 1
    local resolved
    resolved="$(python3 - "$PREBUILT_REPO" "$arch" <<'PY'
import json
import sys
import urllib.request

repo, arch = sys.argv[1], sys.argv[2]
needle = f"linux-x86_64-cuda13-sm{arch}"
api = f"https://api.github.com/repos/{repo}/releases?per_page=20"
req = urllib.request.Request(
    api,
    headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "sparkinfer-bench-scripts",
    },
)
with urllib.request.urlopen(req, timeout=20) as resp:
    releases = json.load(resp)

for release in releases:
    if release.get("draft") or release.get("prerelease"):
        continue
    tag = release.get("tag_name", "")
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name.startswith("sparkinfer-") and needle in name and name.endswith(".tar.gz"):
            print(f"{tag}\t{name}\t{asset['browser_download_url']}")
            raise SystemExit(0)

raise SystemExit(f"no sparkinfer prebuilt asset found for {needle}")
PY
)" || { echo ">> prebuilt release lookup failed" >&2; return 1; }
    IFS=$'\t' read -r tag asset url <<< "$resolved"
    cache_tag="$tag"
  else
    asset="sparkinfer-$tag-linux-x86_64-cuda13-sm${arch}.tar.gz"
    url="https://github.com/$PREBUILT_REPO/releases/download/$tag/$asset"
    cache_tag="$tag"
  fi

  [ -n "$url" ] || url="$PREBUILT_URL"
  [ -n "$asset" ] && [ -n "$url" ] || return 1
  PREBUILT_TAG="$tag"
  PREBUILT_TGZ="$asset"
  PREBUILT_URL="$url"
  PREBUILT_DIR="$ROOT/.prebuilt/${cache_tag}-sm${arch}/sparkinfer-bin"
}

try_prebuilt() {   # download+extract the release bundle; sets SI_BIN/SI_LD; returns 1 if unavailable
  local arch="$1" cache_root tgz_path
  [ "${NO_PREBUILT:-0}" = 1 ] && return 1
  resolve_prebuilt_release "$arch" || return 1
  cache_root="$(dirname "$PREBUILT_DIR")"
  tgz_path="$cache_root/$PREBUILT_TGZ"
  if [ ! -x "$PREBUILT_DIR/bin/qwen3_gguf_bench" ]; then
    command -v curl >/dev/null || return 1
    echo ">> fetching prebuilt $PREBUILT_TGZ from $PREBUILT_TAG ..." >&2
    mkdir -p "$cache_root"
    curl -fsSL "$PREBUILT_URL" -o "$tgz_path" 2>/dev/null || { echo ">> prebuilt download failed" >&2; return 1; }
    tar xzf "$tgz_path" -C "$cache_root" 2>/dev/null || return 1
  fi
  SI_BIN="$PREBUILT_DIR/bin"; SI_LD="$PREBUILT_DIR/lib"; return 0
}

resolve_runner() {   # $1=arch. Prefer an existing local build, else prebuilt, else build from source.
  if [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ]; then SI_BIN="$ROOT/build/runtime"; SI_LD=""; echo ">> using local build" >&2; return; fi
  if try_prebuilt "$1"; then echo ">> using prebuilt binaries (will fall back to source build if incompatible)" >&2; return; fi
  ensure_sparkinfer "$1"; SI_BIN="$ROOT/build/runtime"; SI_LD=""
}

fallback_build() {   # $1=arch. Switch the runner to a fresh source build (prebuilt didn't work here).
  echo ">> prebuilt unusable on this box — building from source ..." >&2
  ensure_sparkinfer "$1"; SI_BIN="$ROOT/build/runtime"; SI_LD=""
}

si_run() {   # si_run <tool> <args...>  — run a sparkinfer binary with the resolved lib path
  if [ -n "$SI_LD" ]; then LD_LIBRARY_PATH="$SI_LD:${LD_LIBRARY_PATH:-}" "$SI_BIN/$1" "${@:2}"
  else "$SI_BIN/$1" "${@:2}"; fi
}

# ---- M1: pin the GPU clocks so tok/s is REPRODUCIBLE run-to-run (not merely same-box-cancelled) ----
# Warmup fixes the cold-clock artifact but leaves boost free to wander with temperature/power, so an
# absolute tok/s isn't reproducible by a third party. Locking the graphics clock to a fixed,
# sustainable value makes the number repeatable; the same-box delta% is unaffected either way. The
# pinned value is reported in the verdict + log so a verifier reproduces at the same clock.
# Best-effort: needs root (eval boxes are root); if the box forbids -lgc, fall back to warmup-only.
GPU_CLOCKS_PINNED=0; PINNED_GCLK=""
# Per card: a dual-GPU (tp=2) box pins EVERY card, each to its own highest supported clock <= cap
# (one card's ladder is not the other's when the pair is mixed). PINNED_GCLK lists them in index
# order; a single-card box behaves exactly as before.
_supported_gclks() { nvidia-smi ${1:+-i "$1"} -q -d SUPPORTED_CLOCKS 2>/dev/null | sed -n 's/.*Graphics *: *\([0-9][0-9]*\) MHz.*/\1/p'; }
pin_clocks() {
  command -v nvidia-smi >/dev/null || return 0
  nvidia-smi -pm 1 >/dev/null 2>&1 || true                      # persistence mode (best-effort)
  local cap="${SPARKINFER_PIN_GCLK_CAP:-2550}" idx tgt pinned=() ok=1
  for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); do
    tgt="${SPARKINFER_PIN_GCLK:-}"
    [ -z "$tgt" ] && tgt=$(_supported_gclks "$idx" | sort -n | awk -v c="$cap" '$1<=c{v=$1} END{print v}')
    if [ -z "$tgt" ]; then
      echo ">> WARN: GPU $idx: no supported graphics clocks found — clocks NOT pinned" >&2; ok=0; break
    fi
    if nvidia-smi -i "$idx" -lgc "$tgt,$tgt" >/dev/null 2>&1; then
      pinned+=("$tgt")
    else
      echo ">> WARN: could not lock GPU $idx clocks (no permission?) — falling back to warmup-only" >&2
      ok=0; break
    fi
  done
  if [ "$ok" = 1 ] && [ "${#pinned[@]}" -gt 0 ]; then
    GPU_CLOCKS_PINNED=1; PINNED_GCLK="$(IFS=,; echo "${pinned[*]}")"
    echo ">> GPU graphics clock(s) pinned to ${PINNED_GCLK} MHz (reproducible tok/s)" >&2
  elif [ "${#pinned[@]}" -gt 0 ]; then
    nvidia-smi -rgc >/dev/null 2>&1 || true                     # all-or-nothing: undo a partial pin
  fi
}
unpin_clocks() {
  [ "$GPU_CLOCKS_PINNED" = 1 ] || return 0
  nvidia-smi -rgc >/dev/null 2>&1 || true
}
