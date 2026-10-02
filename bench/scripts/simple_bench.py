#!/usr/bin/env python3
"""Decode / prefill ladder against a running sparkinfer server (tp=1 or tp=2).

One plain /v1/completions request per measurement; the timings come from the server's own
`usage` block, so no client-side streaming or tokenizer is needed:

  prefill tok/s = prompt_tokens / ttft_ms        (request submit -> first token)
  decode  tok/s = (completion_tokens - 1) / (generation_ms - ttft_ms)
                                                 (tokens after the first / time after the first;
                                                 computed here rather than read from usage.decode_tps,
                                                 which servers before 2026-10-02 divided by the
                                                 whole generation_ms, prefill included)

Three details keep the numbers honest:
  * every prompt is unique (random single-token words, fresh per request), so the server's
    prefix cache never turns a prefill into a cache hit;
  * prompt lengths are exact: filler words are one token each and the server-reported
    prompt_tokens is checked against the target (a mismatch is flagged);
  * there is no ignore_eos, so decode prompts end in an instruction that runs long, and a
    request that stops short of --decode-len is flagged ("short").

Prints one ROW line per measurement (same spirit as HyperQwen's bench/run_benchmarks.sh) and
writes every raw response to --out as JSON lines.

  simple_bench.py [--max-context 16384] [--decode-len 256] [--reps 3] [--conc 1]
                  [--prefill-only | --decode-only] [--decode-prompt prose|count]
"""
import argparse, json, os, random, statistics, sys, threading, time, urllib.request

LADDER = [128, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
# Common English words that are a single token each with a leading space (checked at startup).
WORDS = ("apple river stone cloud green table music light house water paper chair black white "
         "train horse money night sound field glass plant board heart smile dream point story "
         "power space voice watch order level metal north south bread grass sugar")
WORDS = WORDS.split()
SUFFIX = {
    "prose": "\n\nIgnore the list above. Write a long, detailed essay on the history of the Roman "
             "Empire, from its founding to its fall, with many specific events and dates.\n\n",
    "count": "\n\nIgnore the list above. Count from 1 to 3000, separated by commas:",
}


class Client:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip("/"), key

    def _req(self, path, body=None, timeout=3600):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.url + path, data=data, headers=h)
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return json.load(f)

    def info(self):
        return self._req("/v1/info")

    def complete(self, prompt, max_tokens):
        t = time.time()
        d = self._req("/v1/completions", {"prompt": prompt, "max_tokens": max_tokens,
                                          "temperature": 0})
        d["_wall_s"] = time.time() - t
        return d

    def metric(self, name):
        try:
            h = {"Authorization": "Bearer " + self.key} if self.key else {}
            r = urllib.request.Request(self.url + "/metrics", headers=h)
            with urllib.request.urlopen(r, timeout=10) as f:
                for line in f.read().decode().splitlines():
                    if line.startswith(name + " "):
                        return float(line.split()[1])
        except Exception:
            pass
        return None


def filler(n, rng):
    return "".join(" " + rng.choice(WORDS) for _ in range(n))


def run_parallel(fn, args_list):
    out = [None] * len(args_list)
    def work(i, a):
        try:
            out[i] = fn(*a)
        except Exception as e:  # keep the ladder going; report the failure in the row
            out[i] = {"_error": str(e)}
    th = [threading.Thread(target=work, args=(i, a)) for i, a in enumerate(args_list)]
    t = time.time()
    for x in th: x.start()
    for x in th: x.join()
    return out, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8080')}")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--max-context", type=int, default=16384,
                    help="largest prompt length tested, prefill and decode (the server's --ctx must "
                         "also hold a decode prompt + --decode-len)")
    ap.add_argument("--contexts", default="", help="explicit comma list instead of the built-in ladder")
    ap.add_argument("--decode-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3, help="requests per measurement (median reported)")
    ap.add_argument("--conc", default="1", help="comma list of concurrency levels, e.g. 1,2,4")
    ap.add_argument("--decode-prompt", choices=sorted(SUFFIX), default="prose",
                    help="prose: realistic text (DSpark acceptance is modest); count: highly "
                         "predictable (DSpark upper bound)")
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED_BASE", "1000")))
    ap.add_argument("--out", default="bench_results.jsonl")
    a = ap.parse_args()

    c = Client(a.url, a.api_key)
    try:
        info = c.info()
    except Exception as e:
        sys.exit(f"no server at {a.url} ({e})")
    srv_ctx = int(info.get("max_context", 0)) or a.max_context
    max_ctx = min(a.max_context, srv_ctx)
    if a.max_context > srv_ctx:
        print(f"# note: server max_context {srv_ctx} < --max-context {a.max_context}; capped")
    ladder = [int(x) for x in a.contexts.split(",") if x] if a.contexts else LADDER
    concs = [int(x) for x in a.conc.split(",") if x]
    rng = random.Random(a.seed)
    out = open(a.out, "w")
    spec0 = c.metric("sparkinfer_speculative_runs_total")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} server={a.url} model={info.get('model')} "
          f"tp={info.get('tp', 1)} max_context={srv_ctx} (testing up to {max_ctx}) "
          f"link={info.get('link', '-')} speculative={'yes' if spec0 is not None else 'unknown'}")

    # Calibrate: suffix token counts (the server adds no BOS to raw completions, but measure
    # rather than assume) and that a filler word really is one token.
    base = {k: c.complete(v, 1)["usage"]["prompt_tokens"] for k, v in SUFFIX.items()}
    probe = c.complete(filler(200, rng).lstrip(), 1)["usage"]["prompt_tokens"]
    if probe != 200:
        print(f"# warning: 200 filler words tokenized to {probe} tokens; lengths will be approximate")
    # Warmup (first requests after start pay one-time JIT / allocation costs).
    c.complete(filler(1000, rng).lstrip(), 8)
    c.complete(filler(200, rng).lstrip() + SUFFIX[a.decode_prompt], 32)

    def log(kind, target, conc, d):
        d = dict(d); d.pop("choices", None)
        out.write(json.dumps({"kind": kind, "target": target, "conc": conc, **d}) + "\n"); out.flush()

    if not a.decode_only:
        for L in [x for x in ladder if x <= max_ctx]:
            for C in concs:
                reps = max(a.reps, C)
                prompts = [filler(L, rng).lstrip() for _ in range(reps)]
                res, wall = [], 0.0
                for i in range(0, reps, C):   # waves of C concurrent requests
                    r, w = run_parallel(lambda p: c.complete(p, 1), [(p,) for p in prompts[i:i + C]])
                    res += r; wall += w
                for d in res: log("prefill", L, C, d)
                ok = [d for d in res if "_error" not in d]
                if not ok:
                    print(f"ROW prefill len={L} conc={C} | FAILED: {res[0].get('_error')}"); continue
                pt = [d["usage"]["prompt_tokens"] for d in ok]
                tps = [d["usage"]["prompt_tokens"] / (d["usage"]["ttft_ms"] / 1000) for d in ok]
                ttft = [d["usage"]["ttft_ms"] for d in ok]
                flag = "" if all(p == L for p in pt) else f" (len mismatch: {sorted(set(pt))})"
                agg = f" | aggregate={sum(pt) / wall:.0f} tok/s" if C > 1 else ""
                print(f"ROW prefill len={L} conc={C} | {statistics.median(tps):.0f} tok/s/request"
                      f"{agg} | medTTFT={statistics.median(ttft):.0f} ms | n={len(ok)}{flag}")

    if not a.prefill_only:
        sfx = SUFFIX[a.decode_prompt]
        for L in [x for x in ladder if x <= max_ctx and x + a.decode_len <= srv_ctx]:
            n_fill = max(1, L - base[a.decode_prompt])
            for C in concs:
                reps = max(a.reps, C)
                prompts = [filler(n_fill, rng).lstrip() + sfx for _ in range(reps)]
                s0 = c.metric("sparkinfer_speculative_runs_total")
                res, wall = [], 0.0
                for i in range(0, reps, C):
                    r, w = run_parallel(lambda p: c.complete(p, a.decode_len),
                                        [(p,) for p in prompts[i:i + C]])
                    res += r; wall += w
                s1 = c.metric("sparkinfer_speculative_runs_total")
                for d in res: log("decode", L, C, d)
                ok = [d for d in res if "_error" not in d]
                if not ok:
                    print(f"ROW decode ctx={L} conc={C} | FAILED: {res[0].get('_error')}"); continue
                dtps = [(d["usage"]["completion_tokens"] - 1) * 1000.0 /
                        max(d["usage"]["generation_ms"] - d["usage"]["ttft_ms"], 1e-3) for d in ok]
                ct = [d["usage"]["completion_tokens"] for d in ok]
                short = sum(1 for n in ct if n < a.decode_len)
                med = statistics.median(dtps)
                agg = (f" | aggregate={C * med:.1f} tok/s (C x median) e2e={sum(ct) / wall:.1f} tok/s"
                       if C > 1 else "")
                spec = (f" | spec_runs+={s1 - s0:.0f}" if s0 is not None and s1 is not None else "")
                note = f" | short={short}/{len(ok)} (EOS before {a.decode_len})" if short else ""
                print(f"ROW decode ctx={L} conc={C} | {med:.1f} tok/s/request{agg} | "
                      f"medTTFT={statistics.median(d['usage']['ttft_ms'] for d in ok):.0f} ms | "
                      f"out={statistics.median(ct):.0f} tok{spec}{note}")
    print(f"# raw responses in {a.out}")


if __name__ == "__main__":
    main()
