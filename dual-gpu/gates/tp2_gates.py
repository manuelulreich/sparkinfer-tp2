#!/usr/bin/env python3
"""tp=2 eval-gate set (dual-gpu WP-15 (3)): re-baseline, never loosen.

Starts the server itself (several times: deterministic and default mode, with and without the
DSpark draft) and runs, on the dual box:

  correctness (exact, no tolerance)
    lossless   DSpark output is byte-identical to plain decode under the same split
               (SPARKINFER_DETERMINISTIC=1 on both sides), and DSpark actually ran
    determinism  /v1/score of a captured set, scored again: KL 0, top-1 1.0
    batching   the same set scored with 7 requests in flight: identical to sequential
  performance (no-regression tiers against a recorded baseline, default mode)
    decode_tps     greedy decode, 200 tokens, median of 3 (server-reported, TTFT excluded)
    prefill_tps    a 3072-token prompt (128-aligned) and a 3074-token one, max_tokens 1
    dspark_tps     greedy DSpark on a counting and a list prompt (both run to max_tokens)

  tp2_gates.py --model DIR --draft DIR [--record baseline.json] [--baseline baseline.json]

Exit status 0 = every gate passed. --record writes the measured performance as the new baseline
(do that once per box/driver, on a quiet machine, and commit it); --baseline compares against it
with --tol (default 5%) of headroom below the recorded number. Only the standard library is used.
"""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE_GATE = os.path.join(HERE, "score_gate.py")

_story = " ".join(f"Item {i}: the {['red','blue','green'][i % 3]} box holds {i * 7 % 13} marbles."
                  for i in range(60))
PROMPTS = [
    ("count", "Count from 1 to 60, separated by commas:", 200),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", 200),
    ("fact", "The capital of France is", 120),
    ("list", "Here are ten tips for staying productive while working from home:\n1.", 300),
    ("cjk", "请用中文详细介绍一下长城的历史和文化意义。", 300),
    ("long", _story + "\nQuestion: For every item, state how many marbles it holds, one per line.\nAnswer:", 500),
    ("bst", "Write a Python class implementing a binary search tree with insert, delete and "
            "in-order traversal.\n", 400),
]


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Server:
    def __init__(self, a, deterministic, draft, log):
        self.a, self.url = a, f"http://127.0.0.1:{a.port}"
        env = dict(os.environ)
        if deterministic:
            env["SPARKINFER_DETERMINISTIC"] = "1"
        cmd = [os.path.join(a.build, "server", "sparkinfer_server"), "-m", a.model,
               "--tokenizer", os.path.join(a.model, "tokenizer.json"), "--ctx", str(a.ctx),
               "--port", str(a.port), "--host", "127.0.0.1", "--tp", "2", "--devices", a.devices]
        if draft:
            cmd += ["--draft-model", a.draft]
        self.log = open(log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        for _ in range(600):
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited during load (see {log})")
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(60)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()
        time.sleep(3)

    def metric(self, name):
        with urllib.request.urlopen(self.url + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(name + " "):
                    return float(line.split()[1])
        return None

    def generate(self, prompt, max_tokens):
        t = time.time()
        d = post(self.url, "/v1/completions", {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
        return d["choices"][0]["text"], d["usage"], time.time() - t


def gate_correctness(a, res):
    s = Server(a, True, False, os.path.join(a.out, "ar_det.log"))
    try:
        ar = {n: s.generate(p, m)[0] for n, p, m in PROMPTS}
        cap = os.path.join(a.out, "cap.json")
        rc = subprocess.run([sys.executable, SCORE_GATE, "capture", "--url", s.url, "--out", cap]).returncode
        det = par = 1
        if rc == 0:
            subprocess.run([sys.executable, SCORE_GATE, "score", "--url", s.url, "--ref", cap,
                            "--out", os.path.join(a.out, "seq.json")], check=True)
            subprocess.run([sys.executable, SCORE_GATE, "score", "--url", s.url, "--ref", cap,
                            "--parallel", "7", "--out", os.path.join(a.out, "par.json")], check=True)
            exact = ["--min-top1", "1.0", "--max-kl", "0.0"]
            det = subprocess.run([sys.executable, SCORE_GATE, "compare", cap,
                                  os.path.join(a.out, "seq.json")] + exact).returncode
            par = subprocess.run([sys.executable, SCORE_GATE, "compare", cap,
                                  os.path.join(a.out, "par.json")] + exact).returncode
    finally:
        s.stop()
    res["determinism"] = det == 0
    res["batching"] = par == 0
    s = Server(a, True, True, os.path.join(a.out, "dspark_det.log"))
    try:
        sp = {n: s.generate(p, m)[0] for n, p, m in PROMPTS}
        runs = s.metric("sparkinfer_speculative_runs_total") or 0
    finally:
        s.stop()
    diff = [n for n in ar if ar[n] != sp[n]]
    res["lossless"] = not diff and runs > 0
    res["lossless_detail"] = {"prompts": len(ar), "differ": diff, "speculative_runs": runs}


def dtps(usage, wall):
    # The server's own decode rate (excludes time-to-first-token); wall-clock rate as a fallback.
    return usage.get("decode_tps") or usage["completion_tokens"] / wall


def gate_perf(a, res):
    perf = {}
    s = Server(a, False, False, os.path.join(a.out, "ar.log"))
    try:
        s.generate("Warm up.", 8)
        perf["decode_tps"] = statistics.median(
            dtps(u, dt) for _, u, dt in
            (s.generate("Count from 1 to 300, separated by commas:", 200) for _ in range(3)))
        words = ["apple", "river", "stone", "cloud", "green", "table", "music", "light"]
        # Exactly n tokens (each word is one token here; varied per run, never suffixed, because
        # the 128-row alignment of the pass is itself what the two lengths measure).
        for n in (3072, 3074):
            runs = [s.generate("".join(" " + words[(i * 7 + n + k) % len(words)] for i in range(n)), 1)
                    for k in range(2)]
            assert all(u["prompt_tokens"] == n for _, u, _ in runs), "prompt length drifted"
            perf[f"prefill_tps_{n}"] = max(u["prompt_tokens"] / dt for _, u, dt in runs)
    finally:
        s.stop()
    s = Server(a, False, True, os.path.join(a.out, "dspark.log"))
    try:
        s.generate("Warm up.", 8)
        # Prompts whose greedy output reliably runs to max_tokens: a decode rate over an early EOS
        # is mostly time-to-first-token.
        for name, p, m in (PROMPTS[0], PROMPTS[3]):
            runs = [s.generate(p, m) for _ in range(2)]
            perf[f"dspark_tps_{name}"] = max(dtps(u, dt) for _, u, dt in runs)
    finally:
        s.stop()
    res["perf"] = {k: round(v, 1) for k, v in perf.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--build", default="build")
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", default="tp2_gates_out")
    ap.add_argument("--baseline")
    ap.add_argument("--record")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--skip-perf", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "devices": a.devices}
    gate_correctness(a, res)
    if not a.skip_perf:
        gate_perf(a, res)
    ok = res["lossless"] and res["determinism"] and res["batching"]
    if a.baseline and "perf" in res:
        base = json.load(open(a.baseline))["perf"]
        res["perf_regressions"] = {k: (res["perf"].get(k), v) for k, v in base.items()
                                   if res["perf"].get(k) is None or res["perf"][k] < v * (1 - a.tol)}
        ok = ok and not res["perf_regressions"]
    if a.record and "perf" in res:
        json.dump({"perf": res["perf"], "recorded": res["date"],
                   "note": "tp=2 no-regression baseline; re-record per box/driver, never loosen"},
                  open(a.record, "w"), indent=2)
    res["pass"] = ok
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
