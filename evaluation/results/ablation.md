# LGCR Mechanism Ablation

Scenes: 32; raw rows: 2048.

The table reports scene-weighted means. Profile metrics exclude scenes
without a chroma transition. Lower is better except that transition-spread
delta is signed; zero matches the ground-truth width. Edge-MAE confidence
intervals resample scenes and retain all conditions for each sampled scene.

## Reconstruction Error

| method | edge MAE [scene-bootstrap 95% CI] | full MAE | PSNR dB | smooth MAE |
|---|---:|---:|---:|---:|
| lgcr_plain_jinc | 0.025784 [0.021017, 0.030047] | 0.002599 | 51.090 | 0.000000 |
| ablate_similarity | 0.023308 [0.019400, 0.026780] | 0.002325 | 51.204 | 0.000000 |
| ablate_plus_rescue | 0.023421 [0.019500, 0.026923] | 0.002336 | 51.175 | 0.000000 |
| ablate_plus_anisotropy | 0.023448 [0.019521, 0.026946] | 0.002339 | 51.166 | 0.000000 |
| lgcr_algo2 | 0.023283 [0.019231, 0.026881] | 0.002327 | 51.340 | 0.000000 |
| ablate_algo6_ungated | 0.023778 [0.019710, 0.027311] | 0.002398 | 51.194 | 0.000000 |
| ablate_algo6_plus_q | 0.023788 [0.019681, 0.027319] | 0.002402 | 51.226 | 0.000000 |
| lgcr_algo6 | 0.023654 [0.019530, 0.027230] | 0.002390 | 51.340 | 0.000000 |

## Boundary Artifact Measures

| method | bleed profile error | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|
| lgcr_plain_jinc | 0.036571 | 0.066400 | 0.077546 | 1.074609 | 0.006111 |
| ablate_similarity | 0.032965 | 0.082139 | 0.109561 | 0.976058 | 0.004582 |
| ablate_plus_rescue | 0.033139 | 0.083394 | 0.114276 | 0.987376 | 0.004482 |
| ablate_plus_anisotropy | 0.033179 | 0.083694 | 0.115214 | 0.988781 | 0.004468 |
| lgcr_algo2 | 0.032967 | 0.082286 | 0.111584 | 0.976477 | 0.004680 |
| ablate_algo6_ungated | 0.033861 | 0.093288 | 0.075565 | 0.800057 | 0.006818 |
| ablate_algo6_plus_q | 0.033925 | 0.079432 | 0.074750 | 0.820822 | 0.006805 |
| lgcr_algo6 | 0.033737 | 0.079150 | 0.074547 | 0.836910 | 0.006795 |

## Paired Scene-Level Deltas for `lgcr_algo6`

Negative MAE deltas favor LGCR. Confidence intervals resample scenes.

| reference | edge MAE delta [95% CI] | smooth MAE delta [95% CI] | scenes improved |
|---|---:|---:|---:|
| lgcr_plain_jinc | -0.002129 [-0.003326, -0.000959] | -0.000000 [-0.000000, +0.000000] | 59.4% |

## Largest Edge-MAE Regrets vs Plain Jinc

- `test_01019_chevron_soft_chroma` (soft_chroma, D=lanczos, siting=left): +0.006932
- `test_01019_chevron_soft_chroma` (soft_chroma, D=lanczos, siting=center): +0.006838
- `test_01019_chevron_soft_chroma` (soft_chroma, D=bicubic, siting=left): +0.006224
- `test_01019_chevron_soft_chroma` (soft_chroma, D=bicubic, siting=center): +0.006030
- `test_01027_sine_soft_chroma` (soft_chroma, D=lanczos, siting=left): +0.005693

These controlled synthetic results do not establish prevalence or subjective benefit on released animation.
