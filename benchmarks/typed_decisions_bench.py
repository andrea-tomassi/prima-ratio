#!/usr/bin/env python3
"""Typed Decisions (LocalLLaMA/typed-decisions) harness: replay the 400-case test
split against any System One endpoint (semif-server or Jev) and compute the
leaderboard metrics: Acc, Soft acc, KL, TV, Brier, ECE, Score MAE, Within-1, ms/case.

Usage:
  python typed_decisions_bench.py --source server --base http://127.0.0.1:8000 --model semif-gemma4-12b
  python typed_decisions_bench.py --source jev --limit 40        # needs OPENROUTER_API_KEY
"""
import argparse, json, math, os, statistics, sys, time, urllib.request
from collections import defaultdict

PARQUET = os.environ.get("TYPED_DECISIONS_PARQUET",
                           "/home/j3st3r/external-projects/jev/bench/typed-decisions/test-all.parquet")
JEV_API = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
EPS = 1e-6
EPS12 = 1e-12

def load_rows(limit=0):
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    rows = []
    for _, r in df.iterrows():
        rows.append({"id": r["id"], "workflow": r["workflow"],
                     "state": r["state"], "questions": json.loads(r["questions"]),
                     "gold": json.loads(r["gold"])})
    return rows[:limit] if limit else rows

def call_server(base, model, state, questions):
    body = {"state": state, "model": model, "questions": questions}
    req = urllib.request.Request(base.rstrip("/") + "/v1/systemone",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read())
    return d, (time.perf_counter() - t0) * 1000

def call_jev(state, questions, key):
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    req = urllib.request.Request(JEV_API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read())
    return d, (time.perf_counter() - t0) * 1000

def distribution_from_answer(qtype, answer, keys):
    if qtype == "noul":
        p = answer.get("noul")
        if p is None:
            probs = answer.get("probabilities", {})
            p = probs.get("true", probs.get("yes"))
        return [p if k == "true" else 1 - p for k in keys]
    probs = answer.get("probabilities", {})
    return [float(probs[k]) for k in keys]

def compute_metrics(rows, results):
    """Leaderboard metrics. Definitions certified by reproducing the published
    Jev row (8/9 metrics within noise; see BENCHMARKS.md)."""
    dec = []
    case_ms = []
    for row in rows:
        res = results[row["id"]]
        case_ms.append(res["ms"])
        for qname, g in row["gold"].items():
            ans = res["questions"].get(qname)
            if ans is None:
                continue
            keys = list(g["probabilities"].keys())
            p = distribution_from_answer(g["type"], ans, keys)
            gv = [float(g["probabilities"][k]) for k in keys]
            dec.append({"p": p, "g": gv, "keys": keys, "type": g["type"],
                        "gold_i": keys.index(g["label"])})
    def mean(xs):
        return statistics.mean(xs) if xs else None

    acc, soft, kl, tv, brier = [], [], [], [], []
    ece_top, ece_soft = [], []
    mae, w1 = [], []
    all_pairs = []
    correct_pairs = []
    for d in dec:
        p, g = d["p"], d["g"]
        n = len(p)
        am = max(range(n), key=p.__getitem__)
        acc.append(int(am == d["gold_i"]))
        soft.append(g[am])                                   # gold mass at our argmax
        kl.append(sum(gg * math.log(max(gg, EPS12) / max(pp, EPS12)) for pp, gg in zip(p, g)))  # KL(p||g)
        tv.append(0.5 * sum(abs(a - b) for a, b in zip(p, g)))
        brier.append(sum((a - b) ** 2 for a, b in zip(p, g)))
        all_pairs.append((max(p), int(am == d["gold_i"])))   # standard top-label ECE
        correct_pairs.append((max(p), int(g[am] > 0.5)))     # soft-correctness ECE (closest to reference)
        if d["type"] == "score":
            levels = list(range(n))
            eg = sum(i * w for i, w in zip(levels, g))
            ep = sum(i * w for i, w in zip(levels, p))
            mae.append(abs(eg - ep)); w1.append(int(abs(eg - ep) <= 1.0))

    def ece(pairs, bins=10):
        tot = 0.0
        for b in range(bins):
            part = [x for x in pairs if min(bins - 1, int(x[0] * bins)) == b]
            if part:
                tot += abs(sum(ok for _, ok in part) / len(part)
                           - sum(c for c, _ in part) / len(part)) * len(part) / len(pairs)
        return tot

    # pooled macro F1 over all decisions
    labels = sorted({d["keys"][d["gold_i"]] for d in dec} | {d["keys"][max(range(len(d["p"])), key=d["p"].__getitem__)] for d in dec})
    f1s = []
    for L in labels:
        tp = sum(1 for d in dec if d["keys"][d["gold_i"]] == L and d["keys"][max(range(len(d["p"])), key=d["p"].__getitem__)] == L)
        fp = sum(1 for d in dec if d["keys"][d["gold_i"]] != L and d["keys"][max(range(len(d["p"])), key=d["p"].__getitem__)] == L)
        fn = sum(1 for d in dec if d["keys"][d["gold_i"]] == L and d["keys"][max(range(len(d["p"])), key=d["p"].__getitem__)] != L)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))

    return {
        "acc": mean(acc), "soft_acc": mean(soft), "macro_f1": mean(f1s),
        "kl": mean(kl), "tv": mean(tv), "brier": mean(brier),
        "ece_top": ece(all_pairs), "ece_soft": ece(correct_pairs),
        "score_mae": mean(mae), "within1": mean(w1),
        "ms_case_p50": statistics.median(case_ms),
        "decisions": len(acc),
    }, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("server", "jev"), default="server")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="semif-gemma4-12b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-raw", default="")
    args = ap.parse_args()

    rows = load_rows(args.limit)
    key = os.environ.get("OPENROUTER_API_KEY") if args.source == "jev" else None
    results, costs = {}, 0.0
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        if args.source == "server":
            d, ms = call_server(args.base, args.model, row["state"], row["questions"])
            answers = {k: v for k, v in d["answers"].items()}
        else:
            for attempt in range(4):
                try:
                    d, ms = call_jev(row["state"], row["questions"], key); break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(2 ** attempt)
            answers = d["answers"]
            costs += d.get("usage", {}).get("cost", 0)
        results[row["id"]] = {"ms": ms, "questions": answers}
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} · p50 {statistics.median([r['ms'] for r in results.values()]):.0f} ms/case"
                  + (f" · ${costs:.4f}" if args.source == "jev" else ""))
    wall = time.perf_counter() - t0
    metrics, _ = compute_metrics(rows, results)
    print(f"\n=== {args.source}: {len(rows)} casi / {metrics['decisions']} decisioni in {wall:.0f}s ===")
    print(f"Acc {metrics['acc']:.3f} | Soft {metrics['soft_acc']:.3f} | F1 {metrics['macro_f1']:.3f} | "
          f"KL {metrics['kl']:.3f} | TV {metrics['tv']:.3f} | Brier {metrics['brier']:.3f} | "
          f"ECE(top) {metrics['ece_top']:.3f} / ECE(soft) {metrics['ece_soft']:.3f} | "
          f"ScoreMAE {metrics['score_mae']:.3f} | W1 {metrics['within1']:.3f} | p50 {metrics['ms_case_p50']:.0f} ms/case")
    if args.out:
        json.dump(metrics, open(args.out, "w"), indent=1)
        print("output:", args.out)
    if args.save_raw:
        with open(args.save_raw, "w") as f:
            for rid, res in results.items():
                f.write(json.dumps({"id": rid, "ms": res["ms"], "questions": res["questions"]}) + "\n")
        print("raw:", args.save_raw)
    if args.source == "jev":
        print(f"costo totale: ${costs:.4f}")
    print("\nriferimento classifica: Jev 0.727/0.580/1.442/0.251/0.148/0.144/0.391/0.952/710ms · "
          "meraGPT 0.768/0.608/0.096/0.149/0.052/0.180/0.219/0.984/526ms · Prior 0.470/0.430/0.347/0.317/0.189/0.088")

if __name__ == "__main__":
    main()
