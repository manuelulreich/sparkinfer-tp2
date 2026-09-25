#!/usr/bin/env python3
"""Re-runnable mechanical sweep for the dual-GPU (tp=2, P2P) change manifest.

Walks the C++/CUDA sources (and the bench/eval/docker tooling) and records every
site that touches device selection, memory allocation, streams/events/graphs,
inter-GPU communication, or GPU tooling. Findings are merged into
02-change-manifest.json as items with status "candidate", source "scan".

Append-only contract: a re-scan never removes items. If a site disappears from
the source (refactored away), the item stays in the manifest and a human closes
it as "rejected" with a note. This keeps the record auditable end to end.

Usage:
  python3 scan.py            # merge newly found sites into the manifest
  python3 scan.py --dry-run  # print what would be added, write nothing
  python3 scan.py --report   # print the current manifest's scan candidates

Python 3 stdlib only. Run from anywhere; paths are resolved relative to this file.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # dual-gpu/ lives at the repo root
MANIFEST = HERE / "02-change-manifest.json"

# Source trees to sweep. The competitor-benchmark dumps and frozen results trees
# are excluded on purpose: they are third-party logs, not sparkinfer code.
CXX_ROOTS = ["kernels", "moe", "runtime", "server"]
CXX_SUFFIXES = {".cpp", ".cc", ".cu", ".h", ".hpp"}
TOOLING_ROOTS = ["bench/scripts", "eval", "docker"]
TOOLING_SUFFIXES = {".sh", ".py"}

# Categories, highest priority first. A line is filed under the FIRST category
# whose pattern matches (the manifest records one item per file:line, so the
# priority decides which facet is visible).
CATEGORIES = [
    ("comm",
     r"cudaDeviceCanAccessPeer|cudaDeviceEnablePeerAccess|cudaMemcpyPeer|\bP2P\b"),
    ("device-context",
     r"cudaSetDevice|cudaGetDeviceCount|cudaGetDeviceProperties|cudaDeviceGetAttribute|"
     r"cudaDeviceGetPCIBusId|cudaDeviceReset|\bcudaGetDevice\(|\bdevice_id\b"),
    ("stream-graph",
     r"cudaStream|cudaEvent|cudaGraph"),
    ("memory-pool",
     r"cudaMalloc|cudaFree|cudaMemGetInfo|cudaMemset|cudaMemcpy"),
    ("host-staging",
     r"cudaHostAlloc|cudaHostRegister|cudaFreeHost"),
    ("sm-sizing",
     r"multiProcessorCount|\bnum_sms\s*\(\s*\)|\bn_splits\b|adaptive_splits"),
]
TOOLING_CATEGORY = re.compile(r"nvidia-smi|CUDA_VISIBLE_DEVICES|--gpus|nvml")

CLASS_RE = re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)")
# Function-like definition: a line that opens a body ('{') and looks like a
# signature, but is not a call/declaration (no ';' at end) and not a control flow
# keyword. Best-effort: this feeds the heuristic `symbol` field only.
FUNC_RE = re.compile(
    r"^\s*(?:(?:static|inline|virtual|explicit)\s+)?[\w:<>*&~]+\s+([A-Za-z_]\w*)\s*\("
    r"[^;{]*\)\s*(const\s*)?override?\s*\{?\s*$")


def cxx_files():
    out = []
    for root in CXX_ROOTS:
        for p in sorted((REPO_ROOT / root).rglob("*")):
            if p.is_file() and p.suffix in CXX_SUFFIXES and "third_party" not in p.parts:
                out.append(p)
    return out


def tooling_files():
    out = []
    for root in TOOLING_ROOTS:
        for p in sorted((REPO_ROOT / root).rglob("*")):
            if p.is_file() and p.suffix in TOOLING_SUFFIXES:
                out.append(p)
    return out


def heuristic_symbol(lines, idx):
    """Walk backwards for the nearest function-like definition line."""
    for j in range(idx, max(-1, idx - 400), -1):
        m = FUNC_RE.match(lines[j])
        if m:
            return m.group(1)
    return None


def class_context(lines, idx):
    depth, stack = 0, []
    for j in range(idx + 1):
        m = CLASS_RE.match(lines[j])
        if m and not lines[j].rstrip().endswith(";"):
            stack.append(m.group(1))
    return stack[-1] if stack else None


def find_scan_sites():
    """Yield (relpath, absfile, lineno, category, snippet) for every matching line.

    relpath is repo-root-relative so manifest keys are stable across CWDs.
    """
    for f in cxx_files():
        lines = f.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            for cat, pat in CATEGORIES:
                if re.search(pat, line):
                    yield (f.relative_to(REPO_ROOT), f, i + 1, cat, line.strip()[:160])
                    break
    for f in tooling_files():
        lines = f.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if TOOLING_CATEGORY.search(line):
                yield (f.relative_to(REPO_ROOT), f, i + 1, "tooling", line.strip()[:160])


def scan(manifest, dry_run=False):
    # (relpath, lineno) already recorded by ANY item (scan or manual): don't
    # double-file a site a human already owns.
    taken = {(it["file"], it["line"]) for it in manifest["items"]
             if it.get("line") is not None}
    existing_keys = {(it["file"], it["line"], it["category"])
                    for it in manifest["items"] if it.get("source") == "scan"}
    next_id = max((int(it["id"].split("-")[1]) for it in manifest["items"])
                  or [0]) + 1

    added = []
    for rel, f, lineno, category, snippet in find_scan_sites():
        rel = str(rel)
        if (rel, lineno) in taken:
            continue
        key = (rel, lineno, category)
        if key in existing_keys:
            continue
        # Heuristic symbol for C++ files only.
        symbol = None
        if f.suffix in CXX_SUFFIXES:
            lines = f.read_text(errors="replace").splitlines()
            symbol = heuristic_symbol(lines, lineno) or class_context(lines, lineno)
        item = {
            "id": f"CHG-{next_id:04d}",
            "subsystem": subsystem_for(rel),
            "category": category,
            "file": rel,
            "line": lineno,
            "symbol": symbol,
            "snippet": snippet,
            "current": "Auto-located call site (see snippet); audit in P2.",
            "change": None,
            "depends_on": [],
            "verification": "",
            "source": "scan",
            "status": "candidate",
            "notes": "",
        }
        manifest["items"].append(item)
        existing_keys.add(key)
        taken.add((rel, lineno))
        added.append(item)
        next_id += 1

    if not dry_run:
        manifest["meta"]["last_scan"] = {
            "new_items": len(added),
            "total_items": len(manifest["items"]),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return added


def subsystem_for(path):
    p = str(path)
    if p.startswith("moe/"):
        return "moe"
    if p.startswith("server/") or p.startswith("docker/") or p.startswith("eval/"):
        return "server-tooling"
    if p.startswith("bench/"):
        return "bench"
    if p.startswith("kernels/"):
        return "kernels"
    if p.startswith("runtime/"):
        if "/tests/" in p or p.endswith("_test.cpp"):
            return "runtime-tests"
        return "runtime"
    return "other"


def report(manifest):
    from collections import Counter
    c = Counter()
    for it in manifest["items"]:
        if it.get("source") == "scan":
            c[(it["subsystem"], it["category"])] += 1
    print(f"{'subsystem':<16} {'category':<16} n")
    for (sub, cat), n in sorted(c.items()):
        print(f"{sub:<16} {cat:<16} {n}")
    print(f"total scan candidates: {sum(c.values())}")


def main():
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    if "--report" in sys.argv:
        report(manifest)
        return
    dry = "--dry-run" in sys.argv
    n_cxx, n_tool = len(cxx_files()), len(tooling_files())
    print(f"[scan] {n_cxx} C++ files, {n_tool} tooling files under {REPO_ROOT}")
    if n_cxx == 0:
        sys.exit("[scan] ERROR: no C++ sources found -- is dual-gpu/ at the repo root?")
    added = scan(manifest, dry_run=dry)
    if dry:
        for it in added:
            print(f"would add {it['id']}: {it['file']}:{it['line']} [{it['category']}]")
    else:
        print(f"merged {len(added)} new scan candidates "
              f"({len(manifest['items'])} total items)")


if __name__ == "__main__":
    main()
