#!/usr/bin/env python3
"""Render 02-change-manifest.json as human-readable Markdown (02-change-manifest.md).

Stdlib only. Read-only: it never mutates the JSON. Run after edits to the JSON
(or let the CI do it) so the two stay in lockstep.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "02-change-manifest.json"
OUT = HERE / "02-change-manifest.md"


def main():
    m = json.loads(MANIFEST.read_text())
    meta = m["meta"]
    L = []
    a = L.append
    a("# Dual-GPU (tp=2, P2P) — change manifest")
    a("")
    a(f"> **Goal.** {meta['goal']}")
    a(f">")
    a(f"> **Hardware.** {meta['hardware']['gpus']} · {meta['hardware']['arch']} · "
      f"{meta['hardware']['vram_each_gb']} GB each · {meta['hardware']['interconnect']}")
    a(f"> **Weights.** {meta['hardware']['weights']}")
    a(f">")
    a(f"> **Plan.** [01-identification-plan.md](01-identification-plan.md) · "
      f"**Consumed by** {meta['consumed_by']}")
    a(f">")
    a(f"> **Status.** last scan: {m['meta'].get('last_scan', {'new_items': '—'})} · "
      f"{len(m['items'])} items · created {meta['created']}")
    a("")
    a("## How to read this file")
    a("")
    a("| field | meaning |")
    a("|---|---|")
    a("| `id` | `CHG-####`, stable, never reused |")
    for k, v in m["status_lifecycle"].items():
        a(f"| status `{k}` | {v} |")
    a("")
    a("A `change` of `null` means *the required change has not yet been designed* — "
      "that is exactly what the next step (the final implementation plan) is for. "
      "Items with `change: null` and `status: confirmed` are the raw material.")
    a("")

    a("## Subsystems (Phase-P2 audit targets)")
    a("")
    a("| id | name | status |")
    a("|---|---|---|")
    for s in m["subsystems"]:
        a(f"| {s['id']} | {s['name']} | {s['status']} |")
    a("")
    for s in m["subsystems"]:
        a(f"### {s['id']} — {s['name']}")
        a("")
        a(f"**Paths:** {', '.join('`'+p+'`' for p in s['paths'])}")
        a(f"**Why.** {s['why']}")
        a("")
        for c in s["checklist"]:
            a(f"- [ ] {c}")
        a("")

    a("## Items")
    a("")
    order = {st: i for i, st in enumerate(m["status_lifecycle"])}
    items = sorted(m["items"], key=lambda it: it["id"])
    by_status = Counter(it["status"] for it in items)
    a("**By status:** " + " · ".join(f"`{k}` × {v}" for k, v in
                                       sorted(by_status.items(), key=lambda kv: order.get(kv[0], 99))))
    a("")
    a("| id | status | sev | subsystem | category | location | what | current | change (design) | verify |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for it in items:
        loc = it["file"]
        if it.get("line") is not None:
            loc += f":{it['line']}"
        if it.get("symbol"):
            loc += f" · `{it['symbol']}`"
        cur = it["current"].replace("|", "\\|").replace("\n", " ")
        chg = it.get("change")
        chg = "(null — to be designed)" if chg is None else str(chg).replace("|", "\\|").replace("\n", " ")
        a(f"| {it['id']} | {it['status']} | {it.get('severity','—')} | {it['subsystem']} | "
          f"{it['category']} | {loc} | {it.get('snippet') or ''} | {cur} | {chg} | "
          f"{(it.get('verification') or '').replace('|','\\|')[:200]} |")
    a("")
    a("## Notes")
    a("")
    a("- This file is **rendered** from `02-change-manifest.json` (the source of truth). "
      "Edit the JSON, re-run `render.py` — or just let the plan's next step do it.")
    a("- The JSON also carries per-item `depends_on`, `notes`, and `source` (manual vs scan) that "
      "the table above collapses for readability.")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
