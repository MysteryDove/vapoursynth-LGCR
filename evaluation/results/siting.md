# LGCR Siting-Mismatch Results

Scenes: 8; raw rows: 936.

The table reports scene-weighted means. Profile metrics exclude scenes
without a chroma transition. Lower is better except that transition-spread
delta is signed; zero matches the ground-truth width. Edge-MAE confidence
intervals resample scenes and retain all conditions for each sampled scene.

## Reconstruction Error

| method | edge MAE [scene-bootstrap 95% CI] | full MAE | PSNR dB | smooth MAE |
|---|---:|---:|---:|---:|
| zimg_bilinear | 0.038294 [0.026309, 0.046462] | 0.003657 | 46.763 | 0.000000 |
| zimg_bicubic | 0.038608 [0.026361, 0.047280] | 0.003708 | 47.003 | 0.000000 |
| zimg_lanczos3 | 0.040027 [0.027152, 0.049298] | 0.003940 | 47.043 | 0.000000 |
| zimg_spline36 | 0.039556 [0.026869, 0.048634] | 0.003873 | 47.048 | 0.000000 |
| jbu | 0.034730 [0.020078, 0.050133] | 0.003383 | 48.086 | 0.000000 |
| guided_filter | 0.047341 [0.026184, 0.070740] | 0.004889 | 48.385 | 0.000001 |
| wada_ejbf | 0.033669 [0.023018, 0.040694] | 0.003451 | 47.292 | 0.000028 |
| korhonen | 0.031807 [0.020662, 0.040854] | 0.003039 | 47.700 | 0.000000 |
| galosh_form | 0.032267 [0.020945, 0.041912] | 0.003079 | 46.716 | 0.000000 |
| lgcr_plain_jinc | 0.034529 [0.023830, 0.041655] | 0.003361 | 47.187 | 0.000000 |
| lgcr_algo2 | 0.031596 [0.021623, 0.038821] | 0.003052 | 47.879 | 0.000000 |
| lgcr_algo4 | 0.031415 [0.020874, 0.039677] | 0.003058 | 48.052 | 0.000000 |
| lgcr_algo6 | 0.032021 [0.021904, 0.038769] | 0.003118 | 47.531 | 0.000000 |

## Boundary Artifact Measures

| method | bleed profile error | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|
| zimg_bilinear | 0.051741 | 0.618804 | 0.107647 | 1.480867 | 0.000000 |
| zimg_bicubic | 0.052291 | 0.618867 | 0.106727 | 1.126410 | 0.004645 |
| zimg_lanczos3 | 0.055415 | 0.618500 | 0.100682 | 0.985534 | 0.007345 |
| zimg_spline36 | 0.054484 | 0.618503 | 0.100812 | 1.026682 | 0.006468 |
| jbu | 0.049439 | 0.410304 | 0.192008 | 1.041943 | 0.000000 |
| guided_filter | 0.071202 | 0.421475 | 0.087431 | 2.065507 | 0.005274 |
| wada_ejbf | 0.048526 | 0.565955 | 0.103551 | 0.746618 | 0.000000 |
| korhonen | 0.043599 | 0.514987 | 0.152769 | 1.009100 | 0.000000 |
| galosh_form | 0.044621 | 0.486640 | 0.355193 | 0.657223 | 0.000000 |
| lgcr_plain_jinc | 0.047537 | 0.618575 | 0.092870 | 1.056183 | 0.000000 |
| lgcr_algo2 | 0.043708 | 0.541288 | 0.127664 | 0.988900 | 0.000000 |
| lgcr_algo4 | 0.043865 | 0.513005 | 0.141836 | 0.873911 | 0.000338 |
| lgcr_algo6 | 0.044390 | 0.578059 | 0.089579 | 0.879563 | 0.000000 |

## Paired Scene-Level Deltas for `lgcr_algo6`

Negative MAE deltas favor LGCR. Confidence intervals resample scenes.

| reference | edge MAE delta [95% CI] | smooth MAE delta [95% CI] | scenes improved |
|---|---:|---:|---:|
| wada_ejbf | -0.001648 [-0.002509, -0.000811] | -0.000028 [-0.000045, -0.000013] | 87.5% |
| lgcr_plain_jinc | -0.002508 [-0.004575, -0.000692] | -0.000000 [-0.000000, +0.000000] | 62.5% |

## Primary Result by Actual Degradation

| degradation | primary edge MAE | external edge MAE | plain edge MAE |
|---|---:|---:|---:|
| box | 0.032021 | 0.033669 | 0.034529 |

## Primary Result by Scene Condition

| condition | scenes | primary edge MAE | external edge MAE | plain edge MAE |
|---|---:|---:|---:|---:|
| coedge | 3 | 0.035711 | 0.037698 | 0.041748 |
| isoluminant | 1 | 0.039284 | 0.040139 | 0.039284 |
| luma_only | 1 | 0.000000 | 0.000000 | 0.000000 |
| misaligned | 1 | 0.037742 | 0.040330 | 0.039125 |
| ridge | 1 | 0.043376 | 0.045623 | 0.043986 |
| soft_chroma | 1 | 0.028632 | 0.030164 | 0.028594 |

## Siting Assumption Intervention

| method | matched edge MAE | mismatched edge MAE | matched |phase| px | mismatched |phase| px |
|---|---:|---:|---:|---:|
| zimg_bilinear | 0.032266 | 0.041308 | 0.373585 | 0.741414 |
| zimg_bicubic | 0.031600 | 0.042112 | 0.373489 | 0.741556 |
| zimg_lanczos3 | 0.032689 | 0.043695 | 0.372935 | 0.741283 |
| zimg_spline36 | 0.032295 | 0.043187 | 0.372925 | 0.741291 |
| jbu | 0.030020 | 0.037086 | 0.325451 | 0.452731 |
| guided_filter | 0.037102 | 0.052460 | 0.280052 | 0.492187 |
| wada_ejbf | 0.027354 | 0.036826 | 0.358807 | 0.669529 |
| korhonen | 0.025976 | 0.034723 | 0.344961 | 0.600001 |
| galosh_form | 0.027022 | 0.034889 | 0.323513 | 0.568204 |
| lgcr_plain_jinc | 0.027192 | 0.038198 | 0.373045 | 0.741340 |
| lgcr_algo2 | 0.023900 | 0.035443 | 0.327796 | 0.648035 |
| lgcr_algo4 | 0.024738 | 0.034754 | 0.333288 | 0.602863 |
| lgcr_algo6 | 0.024310 | 0.035877 | 0.349506 | 0.692336 |

## Largest Edge-MAE Regrets vs Plain Jinc

- `test_01003_sine_soft_chroma` (soft_chroma, D=box, siting=center): +0.003158
- `test_01003_sine_soft_chroma` (soft_chroma, D=box, siting=left): +0.000447
- `test_01003_sine_soft_chroma` (soft_chroma, D=box, siting=center): +0.000305
- `test_01003_sine_soft_chroma` (soft_chroma, D=box, siting=center): +0.000161
- `test_01005_straight_isoluminant` (isoluminant, D=box, siting=center): +0.000000

These controlled synthetic results do not establish prevalence or subjective benefit on released animation.
