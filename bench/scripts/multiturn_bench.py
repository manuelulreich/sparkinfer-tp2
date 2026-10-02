#!/usr/bin/env python3
"""Multi-turn replay against a running sparkinfer server: the opencode-like workload.

Each conversation is a coding-agent session:
  system prompt (agent instructions, tool schema and the repository's README.md and
  CONTRIBUTING.md as project instructions, ~8k tokens, shared by all conversations)
  -> user task
  -> per turn: the model's answer, then a "tool result" of 1-8k tokens (a real source file of
     this repository, so the drafter sees realistic code), until the prompt reaches
     --max-context tokens or --turns turns.

The assistant messages are the server's own answers, so every turn's prompt extends the last
one and the prefix cache can serve the history (as it does for opencode).

Per turn it reports TTFT, prefilled tokens, cached tokens (usage.prompt_tokens_details), decode
tok/s ((completion - 1) / (generation_ms - ttft_ms), as simple_bench.py), and whether DSpark
produced the tokens (usage.speculative_tokens, servers from 2026-10-02 on; "-" when absent).

  multiturn_bench.py [--conc 1,2,4] [--mode greedy,sampled] [--max-context 60000]
                     [--turns 16] [--turn-len 384] [--out turns.jsonl]

--conc runs that many conversations at once; each concurrency level starts from fresh
conversations (a new random task line), so earlier levels leave nothing useful in the cache
beyond the shared system prompt. Sampled mode sends no temperature: the server's
generation_config defaults apply, as for opencode.

Prints one TURN line per turn and one ROW line per (mode, conc):
  ROW multiturn mode=greedy conc=1 | turns=16 spec=3/16 | decode med=52.1 tok/s |
      TTFT med=812 ms | prefilled=61234 cached=380112 | wall=301.2 s
"""
import argparse, json, os, random, statistics, subprocess, sys, threading, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SYSTEM = """You are a coding agent working in a C++/CUDA inference server repository. You help the \
user by reading files, explaining code, and proposing precise edits. Work step by step: read \
what you need with the tools below, then answer. Keep answers concise and technical. When you \
propose an edit, quote the exact lines to change and give the replacement. Never invent file \
contents; ask for a file with the read tool when you need it.

# Tools

You may call one or more functions to assist with the user query. The available functions are:
"""
TOOLS = [
    ("read", "Read a file from the repository. Returns the file content with line numbers.",
     {"path": ("string", "Path relative to the repository root"),
      "offset": ("integer", "First line to read (1-based)"),
      "limit": ("integer", "Number of lines to read")}),
    ("grep", "Search the repository for a regular expression.",
     {"pattern": ("string", "Regular expression"),
      "path": ("string", "Directory or file to search"),
      "include": ("string", "Glob of files to include, e.g. *.cpp")}),
    ("glob", "List files matching a glob pattern.",
     {"pattern": ("string", "Glob pattern, e.g. runtime/src/**/*.cu")}),
    ("edit", "Replace an exact string in a file.",
     {"path": ("string", "File to edit"), "old_string": ("string", "Exact text to replace"),
      "new_string": ("string", "Replacement text"),
      "replace_all": ("boolean", "Replace every occurrence")}),
    ("write", "Write a whole file.",
     {"path": ("string", "File to write"), "content": ("string", "New content")}),
    ("bash", "Run a shell command in the repository root and return its output.",
     {"command": ("string", "The command"), "timeout": ("integer", "Timeout in milliseconds"),
      "description": ("string", "What the command does, in five to ten words")}),
    ("todowrite", "Replace the task list shown to the user.",
     {"todos": ("array", "Items with content, status (pending/in_progress/completed), priority")}),
    ("webfetch", "Fetch a URL and return it as markdown.",
     {"url": ("string", "The URL"), "format": ("string", "text, markdown or html")}),
]
RULES = """
# Rules
- Prefer reading a file over guessing. Read only what you need.
- Before editing, read the surrounding code; match its style, naming and comment density.
- Run the build after an edit and report errors verbatim.
- Do not add dependencies. Do not reformat code you did not change.
- Answer in plain text; use code blocks for code.
"""
TASKS = [
    "Explain how speculative decoding is wired from the HTTP request down to the draft model, "
    "and point out where memory for the draft is allocated.",
    "I want to understand the prefix cache: where entries are created, how a request finds a hit, "
    "and what is restored. Walk me through it.",
    "Find out how tensor parallelism splits the model across two GPUs and what is exchanged "
    "between the cards per layer.",
    "Review the batch scheduler for fairness problems between long and short requests.",
    "How does the server pick prefill chunk sizes under memory pressure? Is there a cliff?",
    "Trace a chat completion with tools from JSON parsing to the grammar constraint.",
]
FILE_DIRS = ("runtime/src", "runtime/include", "server/src", "server/include", "dual-gpu")


def system_prompt():
    tools = []
    for name, desc, params in TOOLS:
        props = {k: {"type": t, "description": d} for k, (t, d) in params.items()}
        tools.append(json.dumps({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": list(params)[:1]}}}))
    env = ("\n# Environment\nWorking directory: /work/repo\nPlatform: linux\n"
           "Is a git repository: yes\n")
    project = ""
    for doc in ("README.md", "CONTRIBUTING.md"):
        try:
            with open(os.path.join(ROOT, doc), errors="replace") as f:
                project += f"\n# Project instructions ({doc})\n" + f.read()
        except OSError:
            pass
    return SYSTEM + "<tools>\n" + "\n".join(tools) + "\n</tools>\n" + RULES + env + project


def source_files():
    out = subprocess.run(["git", "ls-files", *FILE_DIRS], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    keep = []
    for p in sorted(out):
        if p.endswith((".cpp", ".cu", ".h", ".hpp", ".cuh", ".md")):
            try:
                if os.path.getsize(os.path.join(ROOT, p)) > 4000:
                    keep.append(p)
            except OSError:
                pass
    return keep


def tool_result(path, tokens):
    """`cat -n` of a file, cut to about `tokens` tokens (code runs ~3.3 characters a token)."""
    with open(os.path.join(ROOT, path), errors="replace") as f:
        lines = f.read().splitlines()
    budget, out = int(tokens * 3.3), []
    for i, line in enumerate(lines, 1):
        s = f"{i:6d}\t{line}"
        budget -= len(s) + 1
        if budget < 0:
            break
        out.append(s)
    return f"<tool_result name=\"read\" path=\"{path}\">\n" + "\n".join(out) + "\n</tool_result>"


class Client:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip("/"), key

    def chat(self, messages, max_tokens, temperature, seed):
        body = {"messages": messages, "max_tokens": max_tokens, "seed": seed}
        if temperature is not None:
            body["temperature"] = temperature
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        r = urllib.request.Request(self.url + "/v1/chat/completions",
                                   data=json.dumps(body).encode(), headers=h)
        t = time.time()
        with urllib.request.urlopen(r, timeout=3600) as f:
            d = json.load(f)
        d["_wall_s"] = time.time() - t
        return d


def conversation(c, cid, args, mode, files, sysprompt, rows, lock, t0):
    rng = random.Random(args.seed * 1000 + cid)
    task = TASKS[cid % len(TASKS)]
    messages = [{"role": "system", "content": sysprompt},
                {"role": "user", "content": f"[session {cid}-{rng.randrange(1 << 30):x}] {task}"}]
    temperature = 0 if mode == "greedy" else None
    last_prompt = 0
    for turn in range(args.turns):
        try:
            d = c.chat(messages, args.turn_len, temperature, args.seed + 7919 * cid + turn)
        except Exception as e:
            with lock:
                rows.append({"mode": mode, "conv": cid, "turn": turn, "error": str(e)})
                print(f"TURN {mode} conv={cid} turn={turn} | ERROR {e}", flush=True)
            return
        u = d.get("usage", {})
        msg = d["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or "(no answer)"
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ttft, gen = u.get("ttft_ms", -1.0), u.get("generation_ms", -1.0)
        dtps = (ct - 1) * 1000.0 / max(gen - ttft, 1e-3) if ct > 1 and gen > 0 else 0.0
        spec = u.get("speculative_tokens")
        row = {"mode": mode, "conc": args._conc, "conv": cid, "turn": turn, "prompt_tokens": pt,
               "new_tokens": pt - last_prompt, "cached_tokens": cached,
               "prefilled": pt - cached, "completion_tokens": ct, "ttft_ms": ttft,
               "generation_ms": gen, "decode_tps": dtps, "speculative_tokens": spec,
               "finish": d["choices"][0].get("finish_reason"), "t": time.time() - t0,
               "text": text}
        with lock:
            rows.append(row)
            print(f"TURN {mode} conc={args._conc} conv={cid} turn={turn:2d} | prompt={pt:6d} "
                  f"cached={cached:6d} prefilled={pt - cached:6d} | TTFT={ttft:7.0f} ms | "
                  f"out={ct:4d} decode={dtps:6.1f} tok/s | spec="
                  f"{'-' if spec is None else spec}", flush=True)
        last_prompt = pt
        if pt + ct >= args.max_context:
            return
        messages.append({"role": "assistant", "content": text})
        room = args.max_context - pt - ct - args.turn_len - 200
        want = rng.randint(args.tool_min, args.tool_max)
        if room < 400:
            return
        path = files[rng.randrange(len(files))]
        messages.append({"role": "user",
                         "content": tool_result(path, min(want, room)) +
                         "\n\nContinue with the task. Say what this file tells us, then which file "
                         "you would read next and why."})


def summarize(mode, conc, rows, wall):
    ok = [r for r in rows if "error" not in r]
    later = [r for r in ok if r["turn"] > 0]
    spec_turns = [r for r in ok if r["speculative_tokens"]]
    known = any(r["speculative_tokens"] is not None for r in ok)
    med = lambda xs: statistics.median(xs) if xs else 0.0
    decode = [r["decode_tps"] for r in ok if r["completion_tokens"] > 1]
    spec_d = [r["decode_tps"] for r in spec_turns if r["completion_tokens"] > 1]
    plain_d = [r["decode_tps"] for r in ok if not r["speculative_tokens"] and r["completion_tokens"] > 1]
    out_tok = sum(r["completion_tokens"] for r in ok)
    print(f"ROW multiturn mode={mode} conc={conc} | turns={len(ok)} errors={len(rows) - len(ok)} "
          f"spec={len(spec_turns) if known else '-'}/{len(ok)} | decode med={med(decode):.1f} "
          f"(spec {med(spec_d):.1f}, plain {med(plain_d):.1f}) tok/s | TTFT med={med([r['ttft_ms'] for r in ok]):.0f} "
          f"ms (turn>0 {med([r['ttft_ms'] for r in later]):.0f}) | prefilled="
          f"{sum(r['prefilled'] for r in ok)} cached={sum(r['cached_tokens'] for r in ok)} | "
          f"out={out_tok} e2e={out_tok / wall:.1f} tok/s | wall={wall:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SPARKINFER_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--key", default=os.environ.get("SPARKINFER_API_KEY", ""))
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--mode", default="greedy,sampled")
    ap.add_argument("--max-context", type=int, default=60000,
                    help="stop a conversation once its prompt + answer reaches this many tokens")
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--turn-len", type=int, default=384, help="max_tokens per turn")
    ap.add_argument("--tool-min", type=int, default=1000)
    ap.add_argument("--tool-max", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = source_files()
    if not files:
        sys.exit("no source files found under " + ROOT)
    sysprompt = system_prompt()
    c = Client(args.url, args.key)
    all_rows = []
    run = 0
    for mode in args.mode.split(","):
        for conc in (int(x) for x in args.conc.split(",")):
            args._conc = conc
            rows, lock = [], threading.Lock()
            base = run * 16
            run += 1
            t0 = time.time()
            th = [threading.Thread(target=conversation,
                                   args=(c, base + i, args, mode, files, sysprompt, rows, lock, t0))
                  for i in range(conc)]
            for x in th: x.start()
            for x in th: x.join()
            wall = time.time() - t0
            summarize(mode, conc, rows, wall)
            all_rows += rows
    if args.out:
        with open(args.out, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
