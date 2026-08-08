# LGCR Held-Out Controlled Synthetic Benchmark

Scenes: 32; raw rows: 3328.

The table reports scene-weighted means. Profile metrics exclude scenes
without a chroma transition. Lower is better except that transition-spread
delta is signed; zero matches the ground-truth width. Edge-MAE confidence
intervals resample scenes and retain all conditions for each sampled scene.

## Reconstruction Error

| method | edge MAE [scene-bootstrap 95% CI] | full MAE | PSNR dB | smooth MAE |
|---|---:|---:|---:|---:|
| zimg_bilinear | 0.030343 [0.025018, 0.035095] | 0.002925 | 49.617 | 0.000000 |
| zimg_bicubic | 0.027729 [0.022595, 0.032266] | 0.002708 | 50.692 | 0.000000 |
| zimg_lanczos3 | 0.027418 [0.022310, 0.031964] | 0.002785 | 51.058 | 0.000000 |
| zimg_spline36 | 0.027342 [0.022255, 0.031872] | 0.002751 | 50.996 | 0.000000 |
| jbu | 0.028275 [0.021439, 0.035689] | 0.002756 | 50.863 | 0.000000 |
| guided_filter | 0.038401 [0.028048, 0.049213] | 0.003965 | 51.318 | 0.000001 |
| wada_ejbf | 0.021545 [0.018061, 0.024611] | 0.002301 | 51.399 | 0.000027 |
| korhonen | 0.026185 [0.021473, 0.030628] | 0.002529 | 50.170 | 0.000000 |
| galosh_form | 0.030299 [0.025101, 0.035252] | 0.002928 | 48.073 | 0.000000 |
| lgcr_plain_jinc | 0.025784 [0.021017, 0.030047] | 0.002599 | 51.090 | 0.000000 |
| lgcr_algo2 | 0.023283 [0.019231, 0.026881] | 0.002327 | 51.340 | 0.000000 |
| lgcr_algo4 | 0.022359 [0.018362, 0.026198] | 0.002256 | 51.503 | 0.000000 |
| lgcr_algo6 | 0.023654 [0.019530, 0.027230] | 0.002390 | 51.340 | 0.000000 |

## Boundary Artifact Measures

| method | bleed profile error | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|
| zimg_bilinear | 0.041260 | 0.068325 | 0.092965 | 1.621131 | 0.002001 |
| zimg_bicubic | 0.038090 | 0.067711 | 0.094121 | 1.196302 | 0.006455 |
| zimg_lanczos3 | 0.039106 | 0.066527 | 0.084259 | 1.032090 | 0.008585 |
| zimg_spline36 | 0.038638 | 0.066622 | 0.083192 | 1.077782 | 0.007793 |
| jbu | 0.039462 | 0.174359 | 0.163284 | 0.804446 | 0.001706 |
| guided_filter | 0.056964 | 0.204968 | 0.069187 | 1.609309 | 0.003787 |
| wada_ejbf | 0.031964 | 0.085918 | 0.097616 | 0.767082 | 0.000067 |
| korhonen | 0.035762 | 0.132138 | 0.136112 | 1.038241 | 0.002602 |
| galosh_form | 0.041560 | 0.140985 | 0.413714 | 0.706160 | 0.004451 |
| lgcr_plain_jinc | 0.036571 | 0.066400 | 0.077546 | 1.074609 | 0.006111 |
| lgcr_algo2 | 0.032967 | 0.082286 | 0.111584 | 0.976477 | 0.004680 |
| lgcr_algo4 | 0.032154 | 0.096918 | 0.134899 | 0.753299 | 0.003946 |
| lgcr_algo6 | 0.033737 | 0.079150 | 0.074547 | 0.836910 | 0.006795 |

## Paired Scene-Level Deltas for `lgcr_algo6`

Negative MAE deltas favor LGCR. Confidence intervals resample scenes.

| reference | edge MAE delta [95% CI] | smooth MAE delta [95% CI] | scenes improved |
|---|---:|---:|---:|
| wada_ejbf | +0.002109 [+0.000381, +0.003706] | -0.000027 [-0.000035, -0.000020] | 18.8% |
| lgcr_plain_jinc | -0.002129 [-0.003326, -0.000959] | -0.000000 [-0.000000, +0.000000] | 59.4% |

## Primary Result by Actual Degradation

| degradation | primary edge MAE | external edge MAE | plain edge MAE |
|---|---:|---:|---:|
| bicubic | 0.025155 | 0.019166 | 0.026657 |
| box | 0.021536 | 0.024256 | 0.024343 |
| lanczos | 0.026809 | 0.018361 | 0.027594 |
| triangle | 0.021119 | 0.024397 | 0.024540 |

## Primary Result by Scene Condition

| condition | scenes | primary edge MAE | external edge MAE | plain edge MAE |
|---|---:|---:|---:|---:|
| coedge | 12 | 0.027846 | 0.022416 | 0.034009 |
| isoluminant | 4 | 0.030980 | 0.023912 | 0.030980 |
| luma_only | 4 | 0.000000 | 0.000000 | 0.000000 |
| misaligned | 4 | 0.030563 | 0.028049 | 0.032180 |
| ridge | 4 | 0.032441 | 0.032703 | 0.032581 |
| soft_chroma | 4 | 0.011714 | 0.020448 | 0.008501 |

## Largest Edge-MAE Regrets vs Plain Jinc

- `test_01019_chevron_soft_chroma` (soft_chroma, D=lanczos, siting=left): +0.006932
- `test_01019_chevron_soft_chroma` (soft_chroma, D=lanczos, siting=center): +0.006838
- `test_01019_chevron_soft_chroma` (soft_chroma, D=bicubic, siting=left): +0.006224
- `test_01019_chevron_soft_chroma` (soft_chroma, D=bicubic, siting=center): +0.006030
- `test_01027_sine_soft_chroma` (soft_chroma, D=lanczos, siting=left): +0.005693

These controlled synthetic results do not establish prevalence or subjective benefit on released animation.
