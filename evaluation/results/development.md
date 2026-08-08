# LGCR Development Results

Scenes: 12; raw rows: 468.

The table reports scene-weighted means. Profile metrics exclude scenes
without a chroma transition. Lower is better except that transition-spread
delta is signed; zero matches the ground-truth width. Edge-MAE confidence
intervals resample scenes and retain all conditions for each sampled scene.

## Reconstruction Error

| method | edge MAE [scene-bootstrap 95% CI] | full MAE | PSNR dB | smooth MAE |
|---|---:|---:|---:|---:|
| zimg_bilinear | 0.030039 [0.020818, 0.038281] | 0.002873 | 53.713 | 0.000000 |
| zimg_bicubic | 0.028121 [0.019405, 0.035850] | 0.002713 | 54.386 | 0.000000 |
| zimg_lanczos3 | 0.027663 [0.019083, 0.035219] | 0.002749 | 54.661 | 0.000000 |
| zimg_spline36 | 0.027593 [0.019041, 0.035154] | 0.002721 | 54.609 | 0.000000 |
| jbu | 0.033081 [0.019799, 0.046090] | 0.003216 | 53.428 | 0.000000 |
| guided_filter | 0.046671 [0.026295, 0.067343] | 0.004813 | 53.568 | 0.000000 |
| wada_ejbf | 0.022346 [0.015507, 0.028163] | 0.002351 | 55.148 | 0.000024 |
| korhonen | 0.026022 [0.017153, 0.034053] | 0.002491 | 54.281 | 0.000000 |
| galosh_form | 0.031037 [0.020960, 0.039986] | 0.002972 | 51.950 | 0.000000 |
| lgcr_plain_jinc | 0.025835 [0.017843, 0.032954] | 0.002560 | 54.667 | 0.000000 |
| lgcr_algo2 | 0.024569 [0.016587, 0.031786] | 0.002409 | 55.080 | 0.000000 |
| lgcr_algo4 | 0.024528 [0.016099, 0.032581] | 0.002431 | 55.374 | 0.000000 |
| lgcr_algo6 | 0.023963 [0.016305, 0.030532] | 0.002379 | 55.063 | 0.000000 |

## Boundary Artifact Measures

| method | bleed profile error | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|
| zimg_bilinear | 0.042273 | 0.170784 | 0.102168 | 1.605018 | 0.001529 |
| zimg_bicubic | 0.039974 | 0.170449 | 0.095587 | 1.270311 | 0.005322 |
| zimg_lanczos3 | 0.040492 | 0.170367 | 0.087998 | 1.107996 | 0.007089 |
| zimg_spline36 | 0.040093 | 0.170394 | 0.088303 | 1.149821 | 0.006359 |
| jbu | 0.046285 | 0.277923 | 0.199759 | 1.348900 | 0.001093 |
| guided_filter | 0.069228 | 0.331673 | 0.088280 | 2.115428 | 0.004918 |
| wada_ejbf | 0.033473 | 0.176789 | 0.111930 | 0.796410 | 0.000035 |
| korhonen | 0.036063 | 0.192806 | 0.146288 | 1.123665 | 0.001820 |
| galosh_form | 0.043127 | 0.200466 | 0.553504 | 0.958303 | 0.002617 |
| lgcr_plain_jinc | 0.037651 | 0.170313 | 0.081697 | 1.164033 | 0.003702 |
| lgcr_algo2 | 0.035227 | 0.167471 | 0.110906 | 1.121404 | 0.003070 |
| lgcr_algo4 | 0.035481 | 0.186823 | 0.134174 | 1.041552 | 0.002515 |
| lgcr_algo6 | 0.034824 | 0.180587 | 0.080891 | 1.010110 | 0.003898 |

## Paired Scene-Level Deltas for `lgcr_algo6`

Negative MAE deltas favor LGCR. Confidence intervals resample scenes.

| reference | edge MAE delta [95% CI] | smooth MAE delta [95% CI] | scenes improved |
|---|---:|---:|---:|
| wada_ejbf | +0.001616 [-0.001159, +0.003616] | -0.000024 [-0.000036, -0.000014] | 8.3% |
| lgcr_plain_jinc | -0.001872 [-0.003598, -0.000457] | +0.000000 [+0.000000, +0.000000] | 41.7% |

## Primary Result by Actual Degradation

| degradation | primary edge MAE | external edge MAE | plain edge MAE |
|---|---:|---:|---:|
| bicubic | 0.024936 | 0.017917 | 0.026391 |
| box | 0.025923 | 0.026716 | 0.027832 |
| triangle | 0.021029 | 0.022405 | 0.023282 |

## Primary Result by Scene Condition

| condition | scenes | primary edge MAE | external edge MAE | plain edge MAE |
|---|---:|---:|---:|---:|
| coedge | 3 | 0.026086 | 0.022429 | 0.032497 |
| isoluminant | 2 | 0.031238 | 0.025644 | 0.031238 |
| luma_only | 2 | 0.000000 | 0.000000 | 0.000000 |
| misaligned | 2 | 0.036192 | 0.033966 | 0.037080 |
| ridge | 2 | 0.029478 | 0.027368 | 0.030623 |
| soft_chroma | 1 | 0.015477 | 0.026912 | 0.014647 |

## Largest Edge-MAE Regrets vs Plain Jinc

- `dev_00107_chevron_soft_chroma` (soft_chroma, D=bicubic, siting=left): +0.002038
- `dev_00100_sine_misaligned` (misaligned, D=box, siting=left): +0.001112
- `dev_00107_chevron_soft_chroma` (soft_chroma, D=triangle, siting=left): +0.000749
- `dev_00102_straight_ridge` (ridge, D=bicubic, siting=left): +0.000444
- `dev_00100_sine_misaligned` (misaligned, D=bicubic, siting=left): +0.000027

These controlled synthetic results do not establish prevalence or subjective benefit on released animation.
