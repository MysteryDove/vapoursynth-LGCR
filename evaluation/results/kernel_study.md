# Development-Only LGCR Base-Kernel Screen

Scenes: 12; raw rows: 864.

**Status: EXPLORATORY SCREEN.** This uses the existing development split only.
It cannot replace Jinc3 in the frozen primary analysis, and the existing held-out
split must not be used to confirm a newly selected kernel.

All LGCR runs use strength 0.8, algo6, dense evaluation, and zero hull margin.
Deltas are algo6 minus the signal-only reconstruction with the same base kernel.
Confidence intervals resample scenes; negative values favor algo6.

## Overall Edge Error

| base kernel | plain edge MAE | algo6 edge MAE | algo6-minus-plain [95% CI] | scenes improved |
|---|---:|---:|---:|---:|
| bilinear | 0.030039 | 0.026683 | -0.003356 [-0.006036, -0.001070] | 58.3% |
| bicubic_catrom | 0.026437 | 0.024267 | -0.002170 [-0.004097, -0.000573] | 41.7% |
| bicubic_sharp | 0.025904 | 0.023892 | -0.002012 [-0.003828, -0.000520] | 41.7% |
| bicubic_mitchell | 0.030016 | 0.026529 | -0.003487 [-0.006172, -0.001161] | 50.0% |
| spline16 | 0.025899 | 0.023890 | -0.002009 [-0.003832, -0.000514] | 41.7% |
| spline36 | 0.025147 | 0.023327 | -0.001819 [-0.003441, -0.000473] | 41.7% |
| lanczos2 | 0.026437 | 0.024264 | -0.002173 [-0.004098, -0.000579] | 41.7% |
| lanczos3 | 0.024749 | 0.023043 | -0.001706 [-0.003228, -0.000440] | 41.7% |
| lanczos4 | 0.024696 | 0.023005 | -0.001691 [-0.003175, -0.000452] | 41.7% |
| jinc2 | 0.029405 | 0.026221 | -0.003184 [-0.005713, -0.001018] | 50.0% |
| jinc3 | 0.025835 | 0.023963 | -0.001872 [-0.003598, -0.000457] | 41.7% |
| jinc4 | 0.026512 | 0.024325 | -0.002187 [-0.004041, -0.000635] | 50.0% |

## Screening Decisions

- Lowest development algo6 edge MAE overall: `lanczos4` (0.023005).
- Lowest development non-Jinc algo6 edge MAE: `lanczos4` (0.023005).
- Any confirmatory comparison must freeze the non-Jinc candidate above and use new scenes.

## Algo6 Edge MAE By Degradation

| base kernel | box | triangle | bicubic |
|---|---:|---:|---:|
| bilinear | 0.028665 | 0.025605 | 0.025777 |
| bicubic_catrom | 0.026373 | 0.022009 | 0.024420 |
| bicubic_sharp | 0.026051 | 0.021472 | 0.024154 |
| bicubic_mitchell | 0.028436 | 0.025509 | 0.025643 |
| spline16 | 0.026039 | 0.021455 | 0.024176 |
| spline36 | 0.025745 | 0.020772 | 0.023465 |
| lanczos2 | 0.026381 | 0.022009 | 0.024403 |
| lanczos3 | 0.025564 | 0.020372 | 0.023193 |
| lanczos4 | 0.025781 | 0.020406 | 0.022829 |
| jinc2 | 0.028059 | 0.024830 | 0.025775 |
| jinc3 | 0.025923 | 0.021029 | 0.024936 |
| jinc4 | 0.026645 | 0.021961 | 0.024369 |

## Algo6 Boundary Artifacts

| base kernel | bleed profile | |phase| px | alias px | spread delta px | ringing |
|---|---:|---:|---:|---:|---:|
| bilinear | 0.037360 | 0.185434 | 0.097488 | 1.318252 | 0.002211 |
| bicubic_catrom | 0.034162 | 0.181670 | 0.092453 | 1.089864 | 0.003121 |
| bicubic_sharp | 0.033675 | 0.181117 | 0.089474 | 1.052594 | 0.003230 |
| bicubic_mitchell | 0.037246 | 0.185399 | 0.086631 | 1.300670 | 0.002438 |
| spline16 | 0.033679 | 0.181044 | 0.089274 | 1.051838 | 0.003247 |
| spline36 | 0.033738 | 0.180324 | 0.086631 | 1.001932 | 0.003186 |
| lanczos2 | 0.034159 | 0.181717 | 0.091857 | 1.089678 | 0.003116 |
| lanczos3 | 0.033604 | 0.179862 | 0.086391 | 0.972030 | 0.003245 |
| lanczos4 | 0.034237 | 0.179804 | 0.085054 | 0.966861 | 0.003379 |
| jinc2 | 0.036864 | 0.184971 | 0.086032 | 1.263540 | 0.002736 |
| jinc3 | 0.034824 | 0.180587 | 0.080891 | 1.010110 | 0.003898 |
| jinc4 | 0.037354 | 0.181517 | 0.079681 | 1.069610 | 0.003302 |
