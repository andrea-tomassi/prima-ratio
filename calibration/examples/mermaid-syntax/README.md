# 📐 Worked example: Mermaid syntax validation

A real, end-to-end example of the calibration flow on a task with a **verifiable
ground truth**: deciding whether a Mermaid diagram will render.

| | |
|---|---|
| **State** | A fenced Mermaid diagram block |
| **Question** | Is this diagram syntactically valid? |
| **Options** | `VALID_SYNTAX` · `SYNTAX_ERROR` |
| **Why it's interesting** | The truth is checkable: render the diagram in a real browser and see |

## 🧪 How the dataset was built

**Valid rows** — 50 diagrams extracted from the official Mermaid documentation
(34 pages, 261 candidate blocks, standard diagram types only: flowchart,
sequence, class, state, ER, journey, gantt, pie, mindmap, gitgraph).

**Invalid rows** — 50 corrupted variants across 8 corruption families:
`arrowhead`, `unclosed_bracket`, `keyword_typo`, `colon_gone`,
`mid_line_truncated`, `stray_chars`, `line_delete`, `swap_dir`.

**Labels were verified, not assumed.** Every row was rendered through a
headless-chromium Mermaid runtime (real parse + render). 12 of the 50 corrupted
samples still rendered correctly — usually truncations inside a label text —
and were relabeled `VALID_SYNTAX` rather than kept as errors. Final held-out
set: **62 valid / 38 invalid**. This is the kind of label noise every dataset
has; here it was measured and corrected instead of absorbed by the model.

## 📁 Files

| File | Rows | Mix | Role |
|---|---|---|---|
| `fit60.jsonl` | 60 | 34 valid / 26 invalid | calibration input |
| `eval100.jsonl` | 100 | 62 valid / 38 invalid | held-out evaluation |

Rows carry an `_meta` field with provenance (source diagram type, corruption
kind, render-verification outcome).

## ▶️ Run the calibration

```bash
python3 -c 'import json; rows=[json.loads(l) for l in open("calibration/examples/mermaid-syntax/fit60.jsonl")]; print(json.dumps({"scenario":"mermaid-syntax","dataset":rows}))' \
  | curl -s http://localhost:8000/v1/calibrate -H 'Content-Type: application/json' -d @-
```

(If the server process can read the repo filesystem — e.g. you run it from
source — the same call works with
`"dataset_path": "calibration/examples/mermaid-syntax/fit60.jsonl"` instead of
the inline `dataset`.)

The server scores the 60 rows, fits the temperature, reports the fit numbers,
publishes `semif-gemma4-12b:mermaid-syntax` and hot-reloads it. From then on it
is a normal model name in `GET /v1/models`.

## 📊 Results — held-out `eval100.jsonl`

Calibration fits on `fit60.jsonl`; everything below is measured on the 100 rows
the fit never saw. ECE is computed on the option probabilities, top-label
binning, identical treatment for every system.

| System | Accuracy | ECE | Fitted T |
|---|---|---|---|
| gemma4-12b `:gguf`, vanilla | 77% | 0.174 | — |
| gemma4-12b `:gguf`, calibrated | 77% | **0.133** | 6.09 |
| Qwen3.5-4B `:latest`, vanilla (reference) | 69% | 0.152 | — |
| Qwen3.5-4B `:latest`, calibrated (reference) | 69% | **0.033** | 2.71 |
| Typesafe Jev (SaaS, `typesafe/jev-1.13`, two runs) | 78–80% | 0.048–0.071 | (vendor) |

Fit-set numbers reported by `/v1/calibrate`: gemma4 raw 0.2545 → out-of-fold
0.1064; Qwen3.5-4B raw 0.1545 → out-of-fold 0.142.

## 🎯 What to take away

- **Calibration moves confidence, not accuracy** — the accuracy column is
  identical before and after, by design. What changes is how much you can
  trust the number that comes out, which is what makes confidence gates
  (auto-approve / review) usable.
- **The label verification changed the dataset more than the model did.** 12
  of 50 "broken" samples were not actually broken. Without the render oracle,
  the model would have been trained to call them errors.
- **A frontier SaaS stays ahead by a nose** (higher accuracy *and* tighter
  calibration) — that is the honest baseline. The point of the example is the
  cost profile: 60 labeled rows and one HTTP call, no training, no ML code,
  running on your own GPU.

## 🔁 Reproducing the evaluation

Score a held-out row with the calibrated variant:

```bash
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "semif-gemma4-12b:mermaid-syntax",
  "state": "<the row state, fenced mermaid block>",
  "questions": {"q": {"type": "choice",
    "instructions": "Classify this Mermaid diagram code. Choose VALID_SYNTAX if the code is syntactically correct and renders without errors, or SYNTAX_ERROR if it contains any syntax problem.",
    "criteria": {"VALID_SYNTAX": "The Mermaid code is syntactically correct and renders without errors",
                 "SYNTAX_ERROR": "The Mermaid code contains syntax errors and will fail to render"}}}
}'
```

The `probabilities` in the response are what the ECE table is computed from.
