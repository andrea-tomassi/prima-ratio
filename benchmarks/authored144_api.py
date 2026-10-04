#!/usr/bin/env python3
"""authored144 through the API: balanced accuracy vs gold (same protocol as compare2.py)."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "prima-ratio-clef-flash"
ROWS = "/home/j3st3r/semif-poc/src/benchmarks/data/authored144.jsonl"
OUT = "/home/j3st3r/semif-poc/authored144-api.jsonl"

rows = [json.loads(l) for l in open(ROWS)]
hits, pairs = [], []
t0 = time.perf_counter()
for r in rows:
    body = {"state": r["state"], "model": MODEL,
            "questions": {"q": {"type": "choice", "instructions": r["question"],
                                "criteria": {o["id"]: o["description"] for o in r["options"]}}}}
    req = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    choice = d["answers"]["q"]["choice"]
    gold = [o["id"] for o in r["options"]][r["label"]]
    hits.append(choice == gold)
    pairs.append((r["family"], gold, choice))
wall = time.perf_counter() - t0

acc = sum(hits) / len(hits)
families = {}
for fam, gold, choice in pairs:
    families.setdefault(fam, []).append((gold, choice))
per_fam = {}
for fam, ps in families.items():
    labels = sorted({g for g, _ in ps})
    per_fam[fam] = sum(sum(1 for g, c in ps if g == L and c == L) / max(1, sum(1 for g, _ in ps if g == L)) for L in labels) / len(labels)
bal = sum(per_fam.values()) / len(per_fam)
print(f"model={MODEL} rows={len(rows)} acc={acc:.4f} balanced_acc={bal:.4f} ({wall:.1f}s, {wall/len(rows)*1000:.0f}ms/row)")
for fam, v in sorted(per_fam.items()):
    print(f"  {fam:28s} {v:.3f}")
with open(OUT, "w") as f:
    for r, hit in zip(rows, hits):
        f.write(json.dumps({"id": r["id"], "correct": hit}) + "\n")
