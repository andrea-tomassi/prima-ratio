# Confronto motori — batteria completa (2026-10-01)

Tutte le misure sui container dedicati (:8100) tranne dove il modello era live; T propria di ciascun modello.

| Benchmark | **qat (live)** | **Qwen3.5-9B-Q3** | Qwen3.5-4B | E4B-Q4 † | E2B † | 12B † | Jev/Nimble-9B (pubbl.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nimble macro / micro | 71.0 / 70.9 | **74.2 / 74.0** | 72.8 / 72.9 | 72.9 / 72.6 | 59.1 / 62.0 | 77.1 / 77.7 | 74.8-76.0 / 75.9-77.3 |
| authored144 | 0.868 | **0.875** | 0.819 | 0.882 | 0.806 | 0.944 | — |
| uc fit60 / held40 / batch100 | 0.983/1.000/97 | 0.983/0.975/**99** | 0.967/0.950/98 | 0.983/1.000/97 | 0.683/0.775/99 | —/—/99 | Jev 99 |
| mermaid fit60 / eval100 | 0.650/0.710 | 0.517/0.480 | **0.717/0.700** | 0.633/0.700 | 0.617/0.680 | ~0.79 | — |
| shape777 (accordo) | 0.749 | 0.851 | **0.982** | 0.682 | 0.824 | 0.835 | Jev 0.809 |
| typed-decisions (acc) | **0.657** | 0.618 | 0.601 | 0.658 | — | 0.702 | Jev 0.727 / meraGPT 0.768 |
| typesafe_102 | 0.827/0.192 | 0.817/0.187 | 0.820/0.183 | 0.833/0.175 | — | 0.898/0.140 | Jev 0.883/0.127 |
| perturbations108 | 0.898 | 0.898 | 0.750 | 0.870 | — | — | — |
| Latenza decisione | **54ms** | 100ms | — | 74ms | 44ms | 128-144ms | — |
| typed p50 | **272ms** | 475ms | 543ms | 293ms | — | 691ms | Jev 710ms |
| Nimble mediana/record | **~0.118s** | 0.206s | — | 0.129s | — | 0.29s | — |
| Ctx nativa | 131072 | **262144** | 262144 | 131072 | 131072 | 200K+ | — |
| Peso / VRAM | **4.22GB / 4.45GB** | 5.05GB / ~5.3GB | ~4.4GB | 5.13GB / 5.0GB | 4.7GB | 10.7GB / 13.1GB | — |

## Verdetto
- **qat**: miglior compromesso — 1.8× più veloce, il più piccolo, calibrato meglio (KL 0.426 sul typed);
- **Qwen3.5-9B**: il più accurato dei challenger (Nimble 74.2, a144 0.875, uc 99, perturbations 0.898) ma lento e debole su mermaid;
- **Qwen3.5-4B**: specialista shape777 (0.982, affinità famiglia SemIf) + ctx 256K;
- i due convivono in VRAM (4.45+5.3 = 9.75GB su 16.4) → possibile dual-serving accuratezza/velocità.

Scartati: E2B (debole), LFM2.5 (readout rotto), K2-Horizon (arch non supportata), MiniCPM5-1B (gate fail), MiniCPM5-2B (misto, sotto il qat).
