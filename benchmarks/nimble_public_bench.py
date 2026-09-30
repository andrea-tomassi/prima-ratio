#!/usr/bin/env python3
"""Score the bespokelabsai/nimble public benchmark suite against any System One endpoint.

The 13 subsets of the suite are 3,880 human-labeled records shaped exactly like a
``POST /v1/systemone`` body. This harness reuses the nimble authors' loader, response
validator (``validate_teacher``), scorer (``assess``/``summarize``) and run loop
unchanged; only the HTTP transport points at prima-ratio. Recorded results, the full
method and caveats: ``BENCHMARKS.md``, section 6.

Requirements: a checkout of ``github.com/bespokelabsai/nimble`` at commit ``62076b4``
and the 13 subsets rebuilt with their converters under
``<nimble>/data/public/<subset>/all.jsonl``, verified byte-identical against
``docs/assets/public-benchmarks/subsets/*-manifest.json``.

Two documented adapter notes (semantic-free):
  * probabilities arrive rounded to 4 decimals; they are renormalized by the same sum
    their own ``api_probabilities()`` uses downstream;
  * ``confidence`` on score answers is set only when missing (legacy servers; current
    servers return it).

Usage:
  python nimble_public_bench.py --nimble-repo /path/to/nimble \
      --data /path/to/nimble/data/public --output-dir evaluations/nimble \
      [--endpoint http://HOST:8000/v1/systemone] \
      [--model prima-ratio-gemma4-12b:calibrated] [--concurrency 1] \
      [--subsets vitaminc-dev,boolq,...]
"""

import argparse
import json
import sys
from pathlib import Path

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws",
           "multinli", "civil_comments", "aegis2", "helpsteer2", "summeval-relevance",
           "summeval-consistency", "pubmedqa"]
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/systemone"
DEFAULT_MODEL = "prima-ratio-gemma4-12b:calibrated"


def build_transport(epj, endpoint):
    """HTTP transport to prima-ratio, wrapped with the two documented adapters."""
    base = epj.http_transport("no-auth", url=endpoint, timeout=300)

    def call(payload):
        resp = base(payload)
        ans = resp["answers"]["decision"]
        if ans["type"] != "noul":
            probs = ans["probabilities"]
            total = sum(probs.values())
            if total > 0 and abs(total - 1.0) > 1e-9:
                ans["probabilities"] = {k: v / total for k, v in probs.items()}
        if ans["type"] == "score" and "confidence" not in ans:
            probs = list(ans["probabilities"].values())
            k = len(probs)
            ans["confidence"] = max(0.0, (max(probs) - 1 / k) / (1 - 1 / k)) if k > 1 else 1.0
        return resp

    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nimble-repo", type=Path, required=True,
                        help="checkout of bespokelabsai/nimble (commit 62076b4)")
    parser.add_argument("--data", type=Path, required=True,
                        help="dir holding <subset>/all.jsonl, usually <nimble>/data/public")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--subsets", default=",".join(SUBSETS))
    args = parser.parse_args()

    sys.path.insert(0, str(args.nimble_repo))
    from nimble.evaluation import evaluate_public_jev as epj  # noqa: E402
    epj.API_URL = args.endpoint

    transport = build_transport(epj, args.endpoint)
    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        print(f"=== {subset}", flush=True)
        report = epj.run(args.data / subset / "all.jsonl", args.output_dir / subset,
                         model=args.model, concurrency=args.concurrency, transport=transport,
                         progress=lambda d, t: print(f"  {d}/{t}", flush=True) if d % 50 == 0 or d == t else None)
        print(json.dumps({k: report[k] for k in ("count", "valid", "errors")}), flush=True)


if __name__ == "__main__":
    main()
