# Frozen Supplemental Wada EJBF Holdout

Scenes: 64; raw rows: 3072.

This third synthetic split was frozen after the main and base-kernel
results were known but before these seeds were evaluated. The Wada
comparison is primary within this supplemental study; the hybrid is
exploratory. Neither changes the paper's original primary analysis.

Lower is better; transition-spread delta is best near zero.

| method | edge MAE [95% CI] | bleed profile | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|---:|
| Plain bilinear | 0.030184 [0.026381, 0.033811] | 0.041046 | 0.067279 | 0.093478 | 1.607117 | 0.001996 |
| Wada EJBF | 0.021767 [0.019257, 0.024131] | 0.032326 | 0.079620 | 0.100553 | 0.753627 | 0.000070 |
| Plain Lanczos4 | 0.024640 [0.021492, 0.027645] | 0.036244 | 0.065778 | 0.081758 | 1.041220 | 0.005182 |
| Algo6 Lanczos4 | 0.022730 [0.019850, 0.025472] | 0.033421 | 0.077851 | 0.078539 | 0.848163 | 0.006121 |
| Algo6 Jinc3 | 0.023713 [0.020704, 0.026569] | 0.033734 | 0.078613 | 0.076516 | 0.865489 | 0.006715 |
| Wada + LGCR correction | 0.019990 [0.017341, 0.022563] | 0.029773 | 0.093582 | 0.096819 | 0.552768 | 0.000941 |

## Paired Scene-Level Comparisons

Negative deltas favor the first method. Confidence intervals resample scenes.

| comparison | edge MAE delta [95% CI] | scenes improved |
|---|---:|---:|
| Wada EJBF minus Algo6 Lanczos4 | -0.000963 [-0.002209, +0.000426] | 65.6% |
| Wada EJBF minus Plain bilinear | -0.008418 [-0.010614, -0.006132] | 75.0% |
| Wada EJBF minus Algo6 Jinc3 | -0.001946 [-0.003252, -0.000527] | 70.3% |
| Wada + LGCR correction minus Wada EJBF | -0.001777 [-0.002606, -0.000965] | 45.3% |
| Wada + LGCR correction minus Algo6 Lanczos4 | -0.002740 [-0.004626, -0.000740] | 64.1% |

## Wada EJBF Minus Algo6 Lanczos4 By Degradation

| degradation | Wada EJBF | Algo6 Lanczos4 | paired delta [95% CI] | scenes improved |
|---|---:|---:|---:|---:|
| box | 0.024393 | 0.021811 | +0.002582 [+0.001943, +0.003260] | 12.5% |
| triangle | 0.024659 | 0.021092 | +0.003567 [+0.002600, +0.004613] | 12.5% |
| bicubic | 0.019422 | 0.023399 | -0.003977 [-0.005747, -0.002055] | 73.4% |
| lanczos | 0.018592 | 0.024619 | -0.006027 [-0.008080, -0.003836] | 75.0% |

## Edge MAE By Scene Condition

| condition | scenes | Wada EJBF | Algo6 Lanczos4 | Wada + LGCR | Plain Lanczos4 |
|---|---:|---:|---:|---:|---:|
| coedge | 24 | 0.021695 | 0.025885 | 0.015806 | 0.031441 |
| isoluminant | 8 | 0.024055 | 0.029870 | 0.024055 | 0.029870 |
| luma_only | 8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| misaligned | 8 | 0.029839 | 0.031194 | 0.030278 | 0.031472 |
| ridge | 8 | 0.033349 | 0.032982 | 0.033743 | 0.033137 |
| soft_chroma | 8 | 0.021806 | 0.010140 | 0.024427 | 0.008314 |

The Wada implementation follows the published equation and normalized
parameters but has not been validated bit-for-bit against author code.
The hybrid is a diagnostic composition, not a plugin mode or a selected method.
