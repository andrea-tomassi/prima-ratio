#!/usr/bin/env python3
"""Run SemIf fixtures through Jev (typesafe/jev-1.13 via OpenRouter) with timing capture.

Usage: OPENROUTER_API_KEY=... python jev_bench.py <shape777|authored144|typesafe102> [--limit N]
"""
import argparse, json, os, statistics, sys, time, urllib.request
from collections import Counter, defaultdict

API = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
SEMIF = "/home/j3st3r/external-projects/_tmp/semif-openjev"
OUT_DIR = "/home/j3st3r/external-projects/jev/bench"
FIXTURES = {
    "shape777": f"{SEMIF}/benchmarks/data/shape777.jsonl",
    "authored144": f"{SEMIF}/benchmarks/data/authored144.jsonl",
    "typesafe102": "/home/j3st3r/external-projects/_tmp/ts102/typesafe102.jsonl",
}

def build_question(row):
    """Return (type, criteria) for a fixture row."""
    if row.get("primitive") == "noul":
        crit = {o["id"]: o["description"] for o in row["options"]}
        return "noul", crit
    return "choice", {o["id"]: o["description"] for o in row["options"]}

def call_jev(state, qtype, instructions, criteria, key, timeout=120):
    body = {"model": MODEL, "state": state,
            "questions": {"q": {"type": qtype, "instructions": instructions, "criteria": criteria}}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    return d, (time.perf_counter() - t0) * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", choices=FIXTURES)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY missing (source jev/.env first)")

    rows = [json.loads(l) for l in open(FIXTURES[args.fixture])]
    if args.limit:
        rows = rows[: args.limit]
    preds, costs, tin, tout = [], 0.0, 0, 0
    t_start = time.perf_counter()
    for i, r in enumerate(rows, 1):
        qtype, criteria = build_question(r)
        for attempt in range(4):
            try:
                d, wall_ms = call_jev(r["state"], qtype, r["question"], criteria, key)
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 402):
                    sys.exit(f"Jev auth/quota error {e.code}: {e.read()[:200]}")
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        a = d["answers"]["q"]
        ids = list(criteria.keys())
        if qtype == "noul":
            p = a.get("noul", a.get("probabilities", {}).get("true"))
            probs = [p, 1 - p]
        else:
            probs = [a["probabilities"][k] for k in ids]
        preds.append({"id": r["id"], "option_ids": ids, "probabilities": probs,
                      "wall_ms": round(wall_ms, 1), "usage": d.get("usage", {})})
        costs += d.get("usage", {}).get("cost", 0)
        tin += d.get("usage", {}).get("input_tokens", 0)
        tout += d.get("usage", {}).get("output_tokens", 0)
        if i % 50 == 0 or i == len(rows):
            w = [p["wall_ms"] for p in preds]
            print(f"  {i}/{len(rows)} · p50 {statistics.median(w):.0f}ms · cost ${costs:.4f}")
        time.sleep(0.25)
    wall = time.perf_counter() - t_start
    w = [p["wall_ms"] for p in preds]

    path = f"{OUT_DIR}/jev-{args.fixture}.jsonl"
    with open(path, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    print(f"\n=== {args.fixture}: {len(preds)} chiamate Jev ===")
    print(f"wall totale {wall:.0f}s ({wall/len(preds):.2f}s/call) · API p50 {statistics.median(w):.0f}ms · "
          f"p95 {sorted(w)[int(0.95*len(w))-1]:.0f}ms · token in/out {tin}/{tout} · costo ${costs:.4f}")
    return preds, path

if __name__ == "__main__":
    main()
