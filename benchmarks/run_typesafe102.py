#!/usr/bin/env python3
"""Score the frozen typesafe_public_102 fixture on a semif-server instance and
recompute the published metrics (equal_case_modal_agreement / total_variation)
with the exact protocol of SemIf's benchmarks/evaluate_external.py."""
import json, statistics, sys, time, urllib.request
from collections import defaultdict

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "semif-gemma4-12b"
GOLD = "/home/j3st3r/ts102/typesafe102.jsonl"
COMMITTED_4B = "/home/j3st3r/semif-poc/src/results/raw/predictions/direct-typesafe102.jsonl"
OUT = "/home/j3st3r/ts102/predictions-gemma.jsonl"

rows = [json.loads(l) for l in open(GOLD)]
t0 = time.perf_counter()
preds = []
for r in rows:
    crit = {o["id"]: o["description"] for o in r["options"]}
    qtype = "noul" if r["primitive"] == "noul" else "choice"
    body = {"state": r["state"], "model": MODEL,
            "questions": {"q": {"type": qtype, "instructions": r["question"], "criteria": crit}}}
    req = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    a = d["answers"]["q"]
    if qtype == "noul":
        p = a["noul"]
        ids, probs = ["true", "false"], [p, 1 - p]
    else:
        ids = list(crit.keys())
        probs = [a["probabilities"][i] for i in ids]
    preds.append({"id": r["id"], "option_ids": ids, "probabilities": probs})
wall = time.perf_counter() - t0
with open(OUT, "w") as f:
    for p_ in preds:
        f.write(json.dumps(p_) + "\n")

systems = {"gemma_12b": {p_["id"]: (p_["option_ids"], p_["probabilities"]) for p_ in preds}}
for name in ("typesafe", "opus", "sol"):
    systems[name] = {r["id"]: ([o["id"] for o in r["options"]], r["published_models"][name]["distribution"])
                     for r in rows if name in r.get("published_models", {})}
try:
    c = {json.loads(l)["id"]: json.loads(l) for l in open(COMMITTED_4B)}
    systems["semif_4b"] = {i: (v["option_ids"], v["probabilities"]) for i, v in c.items()}
except FileNotFoundError:
    pass

records = defaultdict(lambda: defaultdict(list))
for row in rows:
    target = row["target_distribution"]
    for name, dist in systems.items():
        if row["id"] not in dist:  # partial coverage (e.g. sol: 101/102)
            continue
        ids, probs = dist[row["id"]]
        if ids != [o["id"] for o in row["options"]]:
            raise SystemExit(f"option order differs for {name} / {row['id']}")
        predicted = max(range(len(probs)), key=probs.__getitem__)
        records[name][row["group_id"]].append({
            "agreement": float(predicted == row["label"]),
            "tv": sum(abs(a - b) for a, b in zip(probs, target)) / 2,
        })

print(f"scored {len(rows)} rows in {wall:.1f}s ({wall/len(rows)*1000:.0f} ms/row) via {MODEL}")
print(f"{'system':12s} {'modal agreement':>16s} {'TV distance':>12s}")
for name, groups in records.items():
    m_ag = statistics.mean(statistics.mean(i["agreement"] for i in v) for v in groups.values())
    m_tv = statistics.mean(statistics.mean(i["tv"] for i in v) for v in groups.values())
    print(f"{name:12s} {m_ag:16.4f} {m_tv:12.4f}   ({sum(len(v) for v in groups.values())} rows / {len(groups)} cases)")
print("\nreference (SemIf repo): published_jev 0.8831 / 0.1268 · semif_4b 0.8453 / 0.1770")
