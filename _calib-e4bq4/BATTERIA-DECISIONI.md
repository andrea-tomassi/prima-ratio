# E4B-Q4 — batteria decisioni (chiusura gap, 2026-10-01)

Modello: `prima-ratio-gemma4-e4b-q4` (live :8000). Metrica shape777 = accordo col riferimento
(4B committed majority), non accuracy.

| Benchmark | E4B-Q4 | Riferimenti |
|---|---|---|
| shape777 (accordo) | **0.6821** (530/777) | E2B 0.8237 · 12B 0.8353 · 27B 0.8443 · **qat 0.7490** |
| typed-decisions (400 casi) | **Acc 0.658** | 12B 0.702 · Jev 0.727 · meraGPT 0.768 |
| ├ calibrata: KL/TV/Brier/ECE | 1.193 / 0.368 / 0.314 / 0.213 | Jev 1.442/0.251/0.148/0.144 · meraGPT 0.096/0.149/0.052/0.180 |
| └ p50 | **293 ms/caso** | 12B 691ms · Jev 710ms · meraGPT 526ms |
| typesafe_102 | **0.8326 / 0.1754** | 12B 0.8978/0.1400 · Jev 0.8831/0.1268 · opus 0.9123/0.1013 |
| perturbations108 (first ever) | **0.8704** (94/108) | — |

perturbations108 per famiglia: evidence_interpretation **1.000** · candidate_selection 0.806 · rule_application 0.806.

Nota shape777: divergenza verificata come genuina (retest singolo identico al batched);
in parte l'E4B concorda col 12B *contro* il riferimento → la metrica misura similarità
alla famiglia, non qualità. Il qat (4.22GB) è più vicino al riferimento (0.749).
