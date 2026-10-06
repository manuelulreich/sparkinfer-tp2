#!/usr/bin/env python3
"""Swift 1.5 (ModelOpt mixed NVFP4/FP8) -> all-NVFP4 ModelOpt checkpoint (plan 11, S1).

The Swift export keeps the 48 Gated-DeltaNet and 16 attention layers' projections in FP8
(per-tensor scale). sparkinfer's tp=2 path is tuned for the layout of
gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090, where every Linear is ModelOpt NVFP4. This tool
requantizes those 208 FP8 Linears to ModelOpt NVFP4 and copies everything else byte for byte:

  .weight          U8       [rows, cols/2]   e2m1 codes, element 2i in the low nibble
  .weight_scale    F8_E4M3  [rows, cols/16]  per-16 block scale
  .weight_scale_2  F32      []               amax / (6 * 448);  W = code * block * scale_2
  .input_scale     F32      []               carried over (sparkinfer never reads it)

Source of the 208 weights (--source):
  bf16  the BF16 originals of ukisai/Swift-1.5-Qwen3.8-27b, fetched tensor by tensor with HTTP
        range requests (~14 GB instead of the 55 GB repository). Default.
  fp8   the FP8 weights of the NVFP4 export itself, dequantized (no download; a second
        quantization on top of FP8).

`mtp.*` is dropped (sparkinfer drafts with DFlash2; the config gets mtp_num_hidden_layers 0).

  python swift_to_nvfp4.py --src ~/models/Swift-1.5-Qwen3.8-27b-NVFP4 --out ~/models/Swift-1.5-Qwen3.8-27b-NVFP4-sparkinfer
  python swift_to_nvfp4.py --src ... --selftest      # quantizer vs ModelOpt's own FFN bytes
"""
import argparse
import multiprocessing as mp
import json
import os
import shutil
import struct
import sys
import threading
import time
import urllib.request

import ml_dtypes
import numpy as np

BF16_REPO = "ukisai/Swift-1.5-Qwen3.8-27b"
BF16_REV = "b4c84d42903a8646b25857eb2827288d92ed85a4"   # weights from the 2026-09-24 release commit

E4M3 = ml_dtypes.float8_e4m3fn
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "I32": 4, "I64": 8}
NP_DT = {"BF16": ml_dtypes.bfloat16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": E4M3}
# e2m1 magnitudes by code, and the decision bounds between neighbouring codes.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)


# ------------------------------------------------------------------ safetensors I/O

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_read(path, base, meta):
    a, b = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        return f.read(b - a)


class RangeSource:
    """BF16 tensors of the source repository, fetched by HTTP range."""

    def __init__(self, repo, rev):
        self.base = f"https://huggingface.co/{repo}/resolve/{rev}/"
        idx = json.loads(self._get("model.safetensors.index.json"))
        self.where = idx["weight_map"]
        self.headers = {}
        self.lock = threading.Lock()

    def _get(self, path, rng=None, tries=5):
        for t in range(tries):
            try:
                req = urllib.request.Request(self.base + path)
                if rng:
                    req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if rng and len(data) != rng[1] - rng[0]:
                    raise IOError(f"short read {len(data)} of {rng[1] - rng[0]}")
                return data
            except Exception as e:  # network hiccup: back off and retry
                if t + 1 == tries:
                    raise
                print(f"  retry {path} ({e})", file=sys.stderr)
                time.sleep(2 + 3 * t)

    def _header(self, shard):
        with self.lock:
            if shard not in self.headers:
                n = struct.unpack("<Q", self._get(shard, (0, 8)))[0]
                self.headers[shard] = (json.loads(self._get(shard, (8, 8 + n))), 8 + n)
            return self.headers[shard]

    def tensor(self, name):
        shard = self.where[name]
        hdr, base = self._header(shard)
        m = hdr[name]
        a, b = m["data_offsets"]
        raw = self._get(shard, (base + a, base + b))
        return np.frombuffer(raw, NP_DT[m["dtype"]]).reshape(m["shape"])


# ------------------------------------------------------------------ NVFP4 (ModelOpt recipe)

def _codes(blk, scale):
    """e2m1 codes (sign in bit 3) and dequantized values of blk [..., 16] under per-block scale."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(scale[..., None] > 0, blk / scale[..., None], 0.0).astype(np.float32)
    mag = np.abs(q)
    code = np.searchsorted(BOUNDS, mag, side="left").astype(np.uint8)   # a tie lands on the lower code
    # Round half to even: a value exactly on bound i (between codes i and i+1) goes to the even code.
    tie = np.zeros(mag.shape, bool)
    for i in (1, 3, 5):   # bounds whose lower neighbour is odd
        tie |= mag == BOUNDS[i]
    code = np.minimum(code + tie.astype(np.uint8), 7)
    deq = E2M1[code] * np.where(np.signbit(q), -1.0, 1.0).astype(np.float32) * scale[..., None]
    code |= (np.signbit(q) & (code > 0)).astype(np.uint8) << 3
    return code, deq


def quant_nvfp4(w, search=4, rows_per_chunk=1024):
    """float32 [rows, cols] -> (packed u8 [rows, cols/2], block e4m3 [rows, cols/16], scale_2 f32).

    ModelOpt's recipe for the tensor-wide scale (amax / (6*448)). The per-16 block scale starts at
    block_amax / 6 like ModelOpt's and then tries the `search` e4m3 codes on either side, keeping
    the one with the least squared error for the block -- measured on Swift's FFN, whose NVFP4
    ModelOpt made: rel RMS error 0.095 for the plain amax scale, 0.0847 for ModelOpt's own bytes,
    0.0833 with search=4.
    """
    w = np.ascontiguousarray(w, np.float32)
    rows, cols = w.shape
    amax = float(np.abs(w).max())
    s2 = np.float32(amax / (6.0 * 448.0)) if amax > 0 else np.float32(1.0)
    packed = np.empty((rows, cols // 2), np.uint8)
    bs_out = np.empty((rows, cols // 16), np.uint8)
    for r0 in range(0, rows, rows_per_chunk):   # chunks keep the candidates' temporaries small
        blk = w[r0:r0 + rows_per_chunk].reshape(-1, cols // 16, 16)
        bamax = np.abs(blk).max(axis=-1)
        nom = np.minimum(bamax / np.float32(6.0) / s2, 448.0).astype(E4M3).view(np.uint8).astype(np.int16)
        best = best_err = best_code = None
        for d in range(-search, search + 1):
            cand = np.clip(nom + d, 0, 126).astype(np.uint8)   # 126 is 448, e4m3fn's largest finite
            code, deq = _codes(blk, cand.view(E4M3).astype(np.float32) * s2)
            err = ((deq - blk) ** 2).sum(axis=-1)
            if best is None:
                best, best_err, best_code = cand, err, code
            else:
                m = err < best_err
                best = np.where(m, cand, best)
                best_err = np.where(m, err, best_err)
                best_code = np.where(m[..., None], code, best_code)
        c = best_code.reshape(-1, cols)
        packed[r0:r0 + rows_per_chunk] = c[:, 0::2] | (c[:, 1::2] << 4)
        bs_out[r0:r0 + rows_per_chunk] = best
    return packed, bs_out.view(E4M3), s2


def dequant_nvfp4(packed, bs8, s2):
    rows = packed.shape[0]
    lo, hi = packed & 15, packed >> 4
    code = np.empty((rows, packed.shape[1] * 2), np.uint8)
    code[:, 0::2], code[:, 1::2] = lo, hi
    v = E2M1[code & 7] * np.where(code & 8, -1.0, 1.0).astype(np.float32)
    v = v.reshape(rows, -1, 16) * (bs8.astype(np.float32) * np.float32(s2))[..., None]
    return v.reshape(rows, -1)


# ------------------------------------------------------------------ conversion

def fp8_linears(src_hdrs):
    """Names (without .weight) of every FP8 Linear in the export."""
    out = []
    for _, (hdr, _) in src_hdrs.items():
        for k, m in hdr.items():
            if k.endswith(".weight") and m.get("dtype") == "F8_E4M3":
                out.append(k[: -len(".weight")])
    return sorted(out)


def plan_shard(hdr, fp8set):
    """Output tensor list for one shard: (name, dtype, shape, producer)."""
    items = []
    for k, m in hdr.items():
        if k == "__metadata__" or k.startswith("mtp."):
            continue
        pre = k.rsplit(".", 1)[0]
        if pre in fp8set:
            if k.endswith(".weight"):
                r, c = m["shape"]
                items.append((pre + ".weight", "U8", [r, c // 2], ("q", pre, "packed")))
                items.append((pre + ".weight_scale", "F8_E4M3", [r, c // 16], ("q", pre, "block")))
                items.append((pre + ".weight_scale_2", "F32", [], ("q", pre, "s2")))
            elif k.endswith(".input_scale"):
                items.append((k, "F32", [], ("in", k)))
            # the FP8 per-tensor .weight_scale has no NVFP4 counterpart
            continue
        items.append((k, m["dtype"], m["shape"], ("copy", k)))
    items.sort(key=lambda t: t[0])
    return items


def nbytes(dtype, shape):
    n = DT_BYTES[dtype]
    for d in shape:
        n *= d
    return n


_W = {}


def _worker_init(source, src, check):
    _W["check"] = check
    if source == "bf16":
        _W["rs"] = RangeSource(BF16_REPO, BF16_REV)
    else:
        idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
        _W["fp8"] = (src, idx, {})


def _load_src(pre):
    if "rs" in _W:
        return _W["rs"].tensor(pre + ".weight").astype(np.float32)
    src, idx, hdrs = _W["fp8"]
    s = idx["weight_map"][pre + ".weight"]
    if s not in hdrs:
        hdrs[s] = st_header(os.path.join(src, s))
    hdr, base = hdrs[s]
    w = np.frombuffer(st_read(os.path.join(src, s), base, hdr[pre + ".weight"]), E4M3)
    w = w.reshape(hdr[pre + ".weight"]["shape"]).astype(np.float32)
    sc = np.frombuffer(st_read(os.path.join(src, s), base, hdr[pre + ".weight_scale"]), np.float32)
    return w * sc[0]


def _worker_quant(pre):
    w = _load_src(pre)
    p, bs8, s2 = quant_nvfp4(w)
    rel = None
    if _W["check"]:
        d = dequant_nvfp4(p, bs8, s2)
        rel = float(np.sqrt(((d - w) ** 2).mean() / max(float((w ** 2).mean()), 1e-30)))
    return pre, p, bs8, s2, rel


def convert(args):
    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    hdrs = {s: st_header(os.path.join(src, s)) for s in shards}
    fp8 = fp8_linears(hdrs)
    fp8set = set(fp8)
    print(f"{len(fp8)} FP8 Linears to requantize, source={args.source}")

    order = []
    plans = {}
    for s in shards:
        plans[s] = plan_shard(hdrs[s][0], fp8set)
        order += [p[3][1] for p in plans[s] if p[3][0] == "q" and p[3][2] == "packed"]

    # Workers fetch and quantize, in output order; the main process only writes. Results are
    # ~0.56 B/weight, so a writer that lags holds little.
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.source, src, args.check))
    results = pool.imap(_worker_quant, order, chunksize=1)

    new_map = {}
    t0 = time.time()
    done = 0
    stats = []
    for s in shards:
        items = plans[s]
        hdr_out, off = {}, 0
        for name, dt, shape, _ in items:
            n = nbytes(dt, shape)
            hdr_out[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
            off += n
        hdr_out["__metadata__"] = {"format": "pt"}
        hjson = json.dumps(hdr_out, separators=(",", ":")).encode()
        hjson += b" " * ((8 - len(hjson) % 8) % 8)
        path = os.path.join(out, s)
        src_hdr, src_base = hdrs[s]
        cache = {}
        with open(path + ".part", "wb") as f, open(os.path.join(src, s), "rb") as fin:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            for name, dt, shape, prod in items:
                if prod[0] == "copy":
                    a, b = src_hdr[prod[1]]["data_offsets"]
                    fin.seek(src_base + a)
                    f.write(fin.read(b - a))
                elif prod[0] == "in":
                    # FP8 activation scale is amax/448; NVFP4's is amax/(6*448). Unused by the runtime.
                    v = np.frombuffer(st_read(os.path.join(src, s), src_base, src_hdr[prod[1]]), np.float32)
                    f.write(np.float32(v[0] / 6.0).tobytes())
                else:
                    pre, part = prod[1], prod[2]
                    if pre not in cache:
                        got, p, bs8, s2, rel = next(results)
                        assert got == pre, (got, pre)
                        if rel is not None:
                            stats.append((pre, rel))
                        cache[pre] = {"packed": p, "block": bs8, "s2": s2}
                        done += 1
                        if done % 16 == 0 or done == len(fp8):
                            el = time.time() - t0
                            print(f"  {done}/{len(fp8)} requantized, {el:.0f} s", flush=True)
                    obj = cache[pre][part]
                    f.write(np.asarray(obj).tobytes())
                    if part == "s2":
                        del cache[pre]
                new_map[name] = s
            assert f.tell() == 8 + len(hjson) + off, "size mismatch"
        os.replace(path + ".part", path)
        print(f"wrote {s} ({(8 + len(hjson) + off) / 1e9:.2f} GB)", flush=True)

    pool.close()
    pool.join()
    total = sum(os.path.getsize(os.path.join(out, s)) for s in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    write_configs(src, out, fp8)
    if stats:
        rels = np.array([r for _, r in stats])
        print(f"requant rel RMS error: mean {rels.mean():.4f}, max {rels.max():.4f} "
              f"({stats[int(rels.argmax())][0]})")
    print(f"done: {total / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {out}")


def write_configs(src, out, fp8):
    for fn in os.listdir(src):
        if fn.endswith((".json", ".jinja", ".txt")) and fn not in (
                "config.json", "hf_quant_config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(src, fn), os.path.join(out, fn))
    for fn in ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(out, fn))
    cfg = json.load(open(os.path.join(src, "config.json")))
    old = cfg.get("quantization_config", {})
    ignore = ["model.language_model.embed_tokens", "model.visual*"]
    for L in range(cfg["text_config"]["num_hidden_layers"]):
        if cfg["text_config"]["layer_types"][L] == "linear_attention":
            ignore += [f"model.language_model.layers.{L}.linear_attn.{n}"
                       for n in ("conv1d", "in_proj_a", "in_proj_b")]
    cfg["quantization_config"] = {
        "config_groups": {"group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"]}},
        "ignore": ignore,
        "quant_algo": "NVFP4",
        "producer": {"name": "sparkinfer swift_to_nvfp4",
                     "from": old.get("producer", {}), "requantized_fp8_linears": len(fp8)},
        "quant_method": "modelopt",
    }
    cfg["text_config"]["mtp_num_hidden_layers"] = 0
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump({"producer": {"name": "modelopt", "version": "sparkinfer-swift_to_nvfp4"},
               "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None,
                                "group_size": 16, "exclude_modules": ignore}},
              open(os.path.join(out, "hf_quant_config.json"), "w"), indent=2)


def selftest(args):
    """Quantize BF16 FFN weights with quant_nvfp4 and compare with ModelOpt's own bytes for them."""
    src = os.path.expanduser(args.src)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    rsrc = RangeSource(BF16_REPO, BF16_REV)
    for pre in ("model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.3.mlp.down_proj",
                "model.language_model.layers.40.mlp.up_proj"):
        s = idx["weight_map"][pre + ".weight"]
        hdr, base = st_header(os.path.join(src, s))
        rd = lambda n, dt: np.frombuffer(st_read(os.path.join(src, s), base, hdr[n]), dt).reshape(hdr[n]["shape"])
        mo_p, mo_b = rd(pre + ".weight", np.uint8), rd(pre + ".weight_scale", E4M3)
        mo_s2 = rd(pre + ".weight_scale_2", np.float32).reshape(())
        w = rsrc.tensor(pre + ".weight").astype(np.float32)
        p, b, s2 = quant_nvfp4(w)
        d_mo = dequant_nvfp4(mo_p, mo_b, mo_s2)
        d_us = dequant_nvfp4(p, b, s2)
        ref = float((w ** 2).mean())
        print(f"{pre}: scale_2 modelopt {float(mo_s2):.6e} ours {float(s2):.6e} | "
              f"block bytes equal {np.mean(mo_b.view(np.uint8) == b.view(np.uint8)) * 100:.3f}% | "
              f"packed bytes equal {np.mean(mo_p == p) * 100:.3f}% | "
              f"rel RMS err modelopt {np.sqrt(((d_mo - w) ** 2).mean() / ref):.4f} "
              f"ours {np.sqrt(((d_us - w) ** 2).mean() / ref):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the downloaded Swift NVFP4 export")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--source", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--workers", type=int, default=3, help="fetch+quantize processes (CPU cores)")
    ap.add_argument("--check", action="store_true", help="report requantization error per tensor")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args)
    else:
        if not args.out:
            ap.error("--out is required")
        convert(args)


if __name__ == "__main__":
    main()
