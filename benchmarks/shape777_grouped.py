#!/usr/bin/env python3
"""shape777 via API, grouped: one request per state (21 questions) — prefix reuse."""
import json, sys, time, urllib.request
from collections import Counter, defaultdict

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "prima-ratio-clef-flash"
ROWS = "/home/j3st3r/semif-poc/src/benchmarks/data/shape777.jsonl"
PRED = "/home/j3st3r/semif-poc/src/results/raw/shape777-direct.predictions.jsonl"
OUT = "/home/j3st3r/semif-poc/shape777-grouped.jsonl"

rows = [json.loads(l) for l in open(ROWS)]
groups = defaultdict(list)
for r in rows:
    groups[r["state"]].append(r)
print(f"righe: {len(rows)} | stati distinti: {len(groups)}")

answers, times, modes = {}, [], Counter()
t0 = time.perf_counter()
for gi, (state, group) in enumerate(groups.items(), 1):
    questions = {r["id"]: {"type": "choice", "instructions": r["question"],
                           "criteria": {o["id"]: o["description"] for o in r["options"]}}
                 for r in group}
    body = {"state": state, "model": MODEL, "questions": questions}
    req = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    times.append(time.perf_counter() - t)
    modes[d["x_prima"]["score_mode"]] += 1
    for qid, a in d["answers"].items():
        answers[qid] = {"choice": a["choice"], "probabilities": a["probabilities"]}
    if gi % 10 == 0:
        print(f"  {gi}/{len(groups)} stati · ultimo {times[-1]:.2f}s")
wall = time.perf_counter() - t0

by_id = defaultdict(list)
for l in open(PRED):
    p = json.loads(l)
    probs = p["probabilities"]
    by_id[p["id"]].append(p["option_ids"][max(range(len(probs)), key=lambda k: probs[k])])
majority = {i: Counter(v).most_common(1)[0][0] for i, v in by_id.items()}

n = len(answers)
agree = sum(1 for i in answers if answers[i]["choice"] == majority.get(i))
print(f"\nn={n} | wall {wall:.1f}s ({wall/60:.2f} min) | media {sum(times)/len(times):.2f}s/richiesta | modes {dict(modes)}")
print(f"agreement vs 4B committed (majority): {agree}/{n} = {agree/n:.4f}  (flips {n-agree})")
print("riferimento: nostro run direct = 0.8353 (128 flips) | exl3-27B = 0.8443 (121 flips)")

try:
    prev = {json.loads(l)["id"]: json.loads(l) for l in open("/home/j3st3r/semif-poc/shape777-gemma-gpu.jsonl")}
    same = 0
    for i, a in answers.items():
        if i in prev:
            p = prev[i]
            top = p["option_ids"][max(range(len(p["probabilities"])), key=lambda k: p["probabilities"][k])]
            same += a["choice"] == top
    print(f"accordo col nostro run direct precedente: {same}/{n} = {same/n:.4f}")
except FileNotFoundError:
    pass

with open(OUT, "w") as f:
    for i, a in answers.items():
        f.write(json.dumps({"id": i, **a}) + "\n")
print("output:", OUT)
