# Frozen Supplemental Kernel Holdout

Scenes: 64; raw rows: 2048.

This holdout was specified after the development screen but before these
seeds were evaluated. It is supplemental and does not replace the main
paper's frozen Jinc3 analysis.

Lower is better; transition-spread delta is best near zero.

| method | edge MAE [95% CI] | bleed profile | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|---:|
| plain_jinc3 | 0.025014 [0.021734, 0.028110] | 0.035844 | 0.067399 | 0.070605 | 1.052517 | 0.005976 |
| algo6_jinc3 | 0.022933 [0.020036, 0.025639] | 0.032929 | 0.081851 | 0.068443 | 0.824147 | 0.006567 |
| plain_lanczos4 | 0.023957 [0.020873, 0.026856] | 0.035591 | 0.067291 | 0.074180 | 1.022345 | 0.005092 |
| algo6_lanczos4 | 0.021988 [0.019232, 0.024507] | 0.032625 | 0.080742 | 0.071482 | 0.802653 | 0.006012 |

## Frozen Paired Comparisons

Negative deltas favor Lanczos4 algo6. Confidence intervals resample scenes.

| reference | edge MAE delta [95% CI] | scenes improved |
|---|---:|---:|
| algo6_jinc3 | -0.000945 [-0.001161, -0.000733] | 81.2% |
| plain_lanczos4 | -0.001969 [-0.002688, -0.001260] | 59.4% |

## Lanczos4 Algo6 Minus Jinc3 Algo6 By Degradation

| degradation | paired edge MAE delta [95% CI] | scenes improved |
|---|---:|---:|
| box | +0.000011 [-0.000132, +0.000157] | 35.9% |
| triangle | -0.000223 [-0.000381, -0.000071] | 57.8% |
| bicubic | -0.001614 [-0.001914, -0.001326] | 85.9% |
| lanczos | -0.001954 [-0.002301, -0.001612] | 87.5% |

## Edge MAE By Scene Condition

| condition | scenes | Lanczos4 algo6 | Jinc3 algo6 | Lanczos4 plain |
|---|---:|---:|---:|---:|
| coedge | 24 | 0.024564 | 0.025580 | 0.030071 |
| isoluminant | 8 | 0.028916 | 0.030487 | 0.028916 |
| luma_only | 8 | 0.000000 | 0.000000 | 0.000000 |
| misaligned | 8 | 0.031849 | 0.033151 | 0.032936 |
| ridge | 8 | 0.029912 | 0.031286 | 0.030332 |
| soft_chroma | 8 | 0.011537 | 0.011802 | 0.009260 |
