#!/usr/bin/env python3
"""Differential teacher-forced correctness gate for tensor parallelism (dual-gpu step 2).

Free-running greedy text is a poor tp=1 vs tp=2 check: one near-tie flips a token and the two
streams never meet again. This gate instead scores the SAME token stream on both servers (the
/v1/score endpoint, teacher-forced) and compares the per-position distributions -- the method
bench/scripts/accuracy_compare_pair.py uses for PR-vs-main.

  # 1. reference (e.g. the RTX 5090, tp=1): generate + score, write the reference file
  score_gate.py capture --url http://HOST:PORT --out ref_5090.json
  # 2. candidate (the 2x 5060 Ti, tp=2): score the reference's own token streams
  score_gate.py score   --url http://127.0.0.1:8091 --ref ref_5090.json --out cand_tp2.json
  # 3. compare (exit 1 on a failed bar)
  score_gate.py compare ref_5090.json cand_tp2.json
  # self-checks on one server: run-twice determinism (start it with SPARKINFER_DETERMINISTIC=1)
  # and continuous batching (--parallel 7 vs the sequential capture)

Only the Python standard library is used. Both servers must serve the same checkpoint.
"""
import argparse
import concurrent.futures
import json
import math
import sys
import urllib.request

TOP_N = 20       # alternatives kept per position
FLOOR = -20.0    # log-prob assumed for a token missing from a top-N list (as in accuracy_compare)
GEN_TOKENS = 96  # greedy completion length captured on the reference

_story = " ".join(f"Item {i}: the {['red', 'blue', 'green'][i % 3]} box holds {i * 7 % 13} marbles."
                  for i in range(120))
PROMPTS = [
    {"name": "fact", "prompt": "The capital of France is"},
    {"name": "code", "prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"},
    {"name": "cjk", "prompt": "请用中文简单介绍一下长城的历史。"},
    {"name": "list", "prompt": "Here are five tips for staying productive while working from home:\n1."},
    {"name": "math", "prompt": "Question: A train travels 120 km in 1.5 hours. What is its average speed?\nAnswer:"},
    {"name": "long", "prompt": _story + "\nQuestion: How many marbles does the box in Item 37 hold? "
                                         "Answer: Item 37 holds"},
    {"name": "chat", "messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]},
]


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def score(url, p, completion_ids):
    body = {"completion_token_ids": completion_ids, "top_logprobs": TOP_N}
    if "messages" in p:
        body["messages"] = p["messages"]
    else:
        body["prompt"] = p["prompt"]
    d = post(url, "/v1/score", body)
    rows = []
    for lp, top in zip(d["logprobs"], d["top_logprobs"]):
        rows.append({"lp": lp, "top": {str(t["token_id"]): t["logprob"] for t in top}})
    return {"token_ids": d["token_ids"], "rows": rows}


def generate_ids(url, p):
    # Greedy text from the reference, then re-tokenised by /v1/score itself (score returns the
    # forced ids), so both servers are scored on one exact id stream.
    if "messages" in p:
        d = post(url, "/v1/chat/completions",
                 {"messages": p["messages"], "max_tokens": GEN_TOKENS, "temperature": 0})
        m = d["choices"][0]["message"]
        text = (m.get("reasoning_content") or "")
        text = ("<think>\n" + text + "\n</think>\n\n" if text else "") + (m.get("content") or "")
    else:
        d = post(url, "/v1/completions", {"prompt": p["prompt"], "max_tokens": GEN_TOKENS, "temperature": 0})
        text = d["choices"][0]["text"]
    body = {"completion": text, "top_logprobs": 1}
    if "messages" in p:
        body["messages"] = p["messages"]
    else:
        body["prompt"] = p["prompt"]
    return post(url, "/v1/score", body)["token_ids"], text


def cmd_capture(a):
    out = {"prompts": []}
    for p in PROMPTS:
        ids, text = generate_ids(a.url, p)
        s = score(a.url, p, ids)
        out["prompts"].append({**p, "completion_ids": ids, "completion_text": text, **s})
        print(f"[capture] {p['name']}: {len(ids)} tokens: {text[:60]!r}", file=sys.stderr)
    json.dump(out, open(a.out, "w"))


def cmd_score(a):
    ref = json.load(open(a.ref))
    # --parallel N submits the prompts concurrently (the continuous-batching check); results are
    # stored in prompt order either way.
    with concurrent.futures.ThreadPoolExecutor(max(1, a.parallel)) as ex:
        scored = list(ex.map(lambda p: score(a.url, p, p["completion_ids"]), ref["prompts"]))
    out = {"prompts": [{"name": p["name"], **s} for p, s in zip(ref["prompts"], scored)]}
    for p, s in zip(ref["prompts"], scored):
        print(f"[score] {p['name']}: {len(s['rows'])} positions", file=sys.stderr)
    json.dump(out, open(a.out, "w"))


def kl(p_top, q_top):
    # KL(P||Q) over the union of both top-N lists, each renormalised over that union.
    keys = set(p_top) | set(q_top)
    lp = {k: p_top.get(k, FLOOR) for k in keys}
    lq = {k: q_top.get(k, FLOOR) for k in keys}
    mp, mq = max(lp.values()), max(lq.values())
    zp = mp + math.log(sum(math.exp(v - mp) for v in lp.values()))
    zq = mq + math.log(sum(math.exp(v - mq) for v in lq.values()))
    return sum(math.exp(lp[k] - zp) * ((lp[k] - zp) - (lq[k] - zq)) for k in keys)


def cmd_compare(a):
    ref = {p["name"]: p for p in json.load(open(a.ref))["prompts"]}
    cand = {p["name"]: p for p in json.load(open(a.cand))["prompts"]}
    ok = True
    tot_n = tot_top1 = 0
    tot_kl = 0.0
    print(f"{'prompt':8s} {'pos':>4s} {'top1':>7s} {'mean|dlp|':>10s} {'meanKL':>9s} {'maxKL':>8s}")
    for name, r in ref.items():
        c = cand.get(name)
        if not c or c["token_ids"] != r["token_ids"]:
            print(f"{name:8s} MISSING or different token stream"); ok = False; continue
        n = len(r["rows"])
        top1 = sum(1 for x, y in zip(r["rows"], c["rows"])
                   if max(x["top"], key=x["top"].get) == max(y["top"], key=y["top"].get))
        dlp = sum(abs(x["lp"] - y["lp"]) for x, y in zip(r["rows"], c["rows"])) / n
        kls = [kl(x["top"], y["top"]) for x, y in zip(r["rows"], c["rows"])]
        mkl = sum(kls) / n
        print(f"{name:8s} {n:4d} {top1 / n:7.3f} {dlp:10.4f} {mkl:9.5f} {max(kls):8.4f}")
        tot_n += n; tot_top1 += top1; tot_kl += sum(kls)
    t1, mk = tot_top1 / max(tot_n, 1), tot_kl / max(tot_n, 1)
    verdict = ok and t1 >= a.min_top1 and mk <= a.max_kl
    print(f"{'ALL':8s} {tot_n:4d} {t1:7.3f} {'':10s} {mk:9.5f}   bars: top1>={a.min_top1} meanKL<={a.max_kl}"
          f" -> {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture"); c.add_argument("--url", required=True); c.add_argument("--out", required=True)
    s = sp.add_parser("score"); s.add_argument("--url", required=True); s.add_argument("--ref", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--parallel", type=int, default=1, help="score N prompts concurrently")
    m = sp.add_parser("compare"); m.add_argument("ref"); m.add_argument("cand")
    m.add_argument("--min-top1", type=float, default=0.95)
    m.add_argument("--max-kl", type=float, default=0.02)
    a = ap.parse_args()
    if a.cmd == "capture": cmd_capture(a)
    elif a.cmd == "score": cmd_score(a)
    else: sys.exit(cmd_compare(a))


if __name__ == "__main__":
    main()
