# Optional BM Refinement Study

This independent study follows `evaluation/bm_study_protocol.md`. All
effects are paired `bm=True` minus `bm=False`; negative error deltas
favor BM. The study does not amend the frozen paper benchmark.

Raw rows: 9,728; wall time: 60.7 seconds.

## Recon Primary Results

| base | observation | edge MAE off | edge MAE on | paired delta [95% CI] | scenes improved | full MAE delta | smooth MAE delta | PSNR delta dB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| algo2_lanczos3 | clean | 0.022608 | 0.022331 | -0.000277 [-0.000325, -0.000227] | 82.8% | -0.000054 | +0.000001 | +0.034 |
| algo2_lanczos3 | q10 | 0.022686 | 0.022405 | -0.000281 [-0.000329, -0.000232] | 89.1% | -0.000055 | +0.000000 | +0.034 |
| algo2_lanczos3 | q10_noise | 0.024718 | 0.024126 | -0.000592 [-0.000653, -0.000538] | 100.0% | -0.001586 | -0.001731 | +0.943 |
| algo6_jinc3 | clean | 0.022933 | 0.022656 | -0.000278 [-0.000334, -0.000218] | 78.1% | -0.000055 | +0.000001 | +0.022 |
| algo6_jinc3 | q10 | 0.023017 | 0.022738 | -0.000279 [-0.000335, -0.000219] | 82.8% | -0.000056 | +0.000000 | +0.022 |
| algo6_jinc3 | q10_noise | 0.025878 | 0.025300 | -0.000578 [-0.000640, -0.000517] | 98.4% | -0.001455 | -0.001587 | +0.909 |
| algo6_lanczos4 | clean | 0.021988 | 0.021721 | -0.000267 [-0.000318, -0.000215] | 79.7% | -0.000075 | +0.000001 | +0.034 |
| algo6_lanczos4 | q10 | 0.022069 | 0.021800 | -0.000269 [-0.000319, -0.000216] | 84.4% | -0.000076 | +0.000000 | +0.034 |
| algo6_lanczos4 | q10_noise | 0.024818 | 0.024227 | -0.000591 [-0.000657, -0.000531] | 100.0% | -0.001646 | -0.001793 | +1.004 |

## Recon Boundary Effects

| base | observation | bleed delta | phase delta px | alias delta px | spread delta px | ringing delta |
|---|---|---:|---:|---:|---:|---:|
| algo2_lanczos3 | clean | -0.000744 | +0.000099 | +0.000413 | -0.008607 | -0.000288 |
| algo2_lanczos3 | q10 | -0.000751 | +0.000090 | +0.000412 | -0.008677 | -0.000292 |
| algo2_lanczos3 | q10_noise | -0.001534 | +0.000046 | +0.000194 | -0.007767 | -0.000756 |
| algo6_jinc3 | clean | -0.000755 | -0.000033 | +0.001224 | -0.005972 | -0.000330 |
| algo6_jinc3 | q10 | -0.000757 | -0.000037 | +0.001217 | -0.005950 | -0.000331 |
| algo6_jinc3 | q10_noise | -0.001442 | -0.000070 | +0.000779 | -0.004611 | -0.000787 |
| algo6_lanczos4 | clean | -0.000993 | -0.000017 | +0.000627 | -0.005888 | -0.000411 |
| algo6_lanczos4 | q10 | -0.000997 | -0.000017 | +0.000614 | -0.005875 | -0.000414 |
| algo6_lanczos4 | q10_noise | -0.001680 | -0.000024 | +0.000203 | -0.005018 | -0.000895 |

## Recon Edge-MAE Delta By Degradation

| base | observation | degradation | paired delta [95% CI] | scenes improved |
|---|---|---|---:|---:|
| algo2_lanczos3 | clean | box | -0.000126 [-0.000153, -0.000098] | 81.2% |
| algo2_lanczos3 | clean | triangle | -0.000173 [-0.000209, -0.000138] | 84.4% |
| algo2_lanczos3 | clean | bicubic | -0.000426 [-0.000500, -0.000350] | 82.8% |
| algo2_lanczos3 | clean | lanczos | -0.000382 [-0.000449, -0.000311] | 82.8% |
| algo2_lanczos3 | q10 | box | -0.000135 [-0.000163, -0.000108] | 85.9% |
| algo2_lanczos3 | q10 | triangle | -0.000184 [-0.000219, -0.000149] | 89.1% |
| algo2_lanczos3 | q10 | bicubic | -0.000425 [-0.000499, -0.000348] | 89.1% |
| algo2_lanczos3 | q10 | lanczos | -0.000381 [-0.000449, -0.000310] | 87.5% |
| algo2_lanczos3 | q10_noise | box | -0.000607 [-0.000667, -0.000552] | 100.0% |
| algo2_lanczos3 | q10_noise | triangle | -0.000610 [-0.000671, -0.000556] | 100.0% |
| algo2_lanczos3 | q10_noise | bicubic | -0.000599 [-0.000665, -0.000539] | 100.0% |
| algo2_lanczos3 | q10_noise | lanczos | -0.000552 [-0.000620, -0.000492] | 100.0% |
| algo6_jinc3 | clean | box | -0.000074 [-0.000103, -0.000042] | 68.8% |
| algo6_jinc3 | clean | triangle | -0.000114 [-0.000144, -0.000083] | 78.1% |
| algo6_jinc3 | clean | bicubic | -0.000473 [-0.000567, -0.000375] | 78.1% |
| algo6_jinc3 | clean | lanczos | -0.000450 [-0.000539, -0.000356] | 81.2% |
| algo6_jinc3 | q10 | box | -0.000080 [-0.000110, -0.000048] | 78.1% |
| algo6_jinc3 | q10 | triangle | -0.000119 [-0.000149, -0.000088] | 82.8% |
| algo6_jinc3 | q10 | bicubic | -0.000470 [-0.000563, -0.000371] | 81.2% |
| algo6_jinc3 | q10 | lanczos | -0.000447 [-0.000537, -0.000354] | 84.4% |
| algo6_jinc3 | q10_noise | box | -0.000558 [-0.000618, -0.000500] | 100.0% |
| algo6_jinc3 | q10_noise | triangle | -0.000585 [-0.000646, -0.000528] | 100.0% |
| algo6_jinc3 | q10_noise | bicubic | -0.000597 [-0.000665, -0.000529] | 98.4% |
| algo6_jinc3 | q10_noise | lanczos | -0.000570 [-0.000639, -0.000502] | 98.4% |
| algo6_lanczos4 | clean | box | -0.000142 [-0.000177, -0.000107] | 75.0% |
| algo6_lanczos4 | clean | triangle | -0.000137 [-0.000163, -0.000110] | 82.8% |
| algo6_lanczos4 | clean | bicubic | -0.000415 [-0.000493, -0.000333] | 79.7% |
| algo6_lanczos4 | clean | lanczos | -0.000376 [-0.000448, -0.000302] | 79.7% |
| algo6_lanczos4 | q10 | box | -0.000148 [-0.000183, -0.000112] | 79.7% |
| algo6_lanczos4 | q10 | triangle | -0.000144 [-0.000170, -0.000117] | 87.5% |
| algo6_lanczos4 | q10 | bicubic | -0.000412 [-0.000493, -0.000330] | 84.4% |
| algo6_lanczos4 | q10 | lanczos | -0.000373 [-0.000444, -0.000298] | 82.8% |
| algo6_lanczos4 | q10_noise | box | -0.000597 [-0.000661, -0.000537] | 100.0% |
| algo6_lanczos4 | q10_noise | triangle | -0.000634 [-0.000697, -0.000579] | 100.0% |
| algo6_lanczos4 | q10_noise | bicubic | -0.000585 [-0.000654, -0.000521] | 100.0% |
| algo6_lanczos4 | q10_noise | lanczos | -0.000547 [-0.000620, -0.000481] | 100.0% |

## Recon Edge-MAE Delta By Scene Condition

| base | observation | condition | paired delta [95% CI] | scenes improved |
|---|---|---|---:|---:|
| algo2_lanczos3 | clean | coedge | -0.000371 [-0.000436, -0.000303] | 100.0% |
| algo2_lanczos3 | clean | isoluminant | -0.000344 [-0.000433, -0.000271] | 100.0% |
| algo2_lanczos3 | clean | luma_only | +0.000000 [+0.000000, +0.000000] | 0.0% |
| algo2_lanczos3 | clean | misaligned | -0.000365 [-0.000412, -0.000293] | 100.0% |
| algo2_lanczos3 | clean | ridge | -0.000331 [-0.000437, -0.000211] | 100.0% |
| algo2_lanczos3 | clean | soft_chroma | -0.000061 [-0.000148, +0.000021] | 62.5% |
| algo2_lanczos3 | q10 | coedge | -0.000377 [-0.000442, -0.000310] | 100.0% |
| algo2_lanczos3 | q10 | isoluminant | -0.000349 [-0.000432, -0.000279] | 100.0% |
| algo2_lanczos3 | q10 | luma_only | +0.000000 [-0.000000, +0.000000] | 37.5% |
| algo2_lanczos3 | q10 | misaligned | -0.000376 [-0.000429, -0.000307] | 100.0% |
| algo2_lanczos3 | q10 | ridge | -0.000329 [-0.000433, -0.000210] | 100.0% |
| algo2_lanczos3 | q10 | soft_chroma | -0.000065 [-0.000151, +0.000017] | 75.0% |
| algo2_lanczos3 | q10_noise | coedge | -0.000584 [-0.000624, -0.000544] | 100.0% |
| algo2_lanczos3 | q10_noise | isoluminant | -0.000582 [-0.000621, -0.000543] | 100.0% |
| algo2_lanczos3 | q10_noise | luma_only | -0.001071 [-0.001226, -0.000915] | 100.0% |
| algo2_lanczos3 | q10_noise | misaligned | -0.000568 [-0.000602, -0.000546] | 100.0% |
| algo2_lanczos3 | q10_noise | ridge | -0.000504 [-0.000565, -0.000446] | 100.0% |
| algo2_lanczos3 | q10_noise | soft_chroma | -0.000257 [-0.000318, -0.000194] | 100.0% |
| algo6_jinc3 | clean | coedge | -0.000392 [-0.000459, -0.000324] | 100.0% |
| algo6_jinc3 | clean | isoluminant | -0.000355 [-0.000437, -0.000286] | 100.0% |
| algo6_jinc3 | clean | luma_only | +0.000000 [+0.000000, +0.000000] | 0.0% |
| algo6_jinc3 | clean | misaligned | -0.000407 [-0.000491, -0.000315] | 100.0% |
| algo6_jinc3 | clean | ridge | -0.000356 [-0.000472, -0.000218] | 100.0% |
| algo6_jinc3 | clean | soft_chroma | +0.000075 [+0.000019, +0.000134] | 25.0% |
| algo6_jinc3 | q10 | coedge | -0.000393 [-0.000460, -0.000326] | 100.0% |
| algo6_jinc3 | q10 | isoluminant | -0.000358 [-0.000438, -0.000288] | 100.0% |
| algo6_jinc3 | q10 | luma_only | +0.000000 [-0.000000, +0.000000] | 37.5% |
| algo6_jinc3 | q10 | misaligned | -0.000410 [-0.000491, -0.000319] | 100.0% |
| algo6_jinc3 | q10 | ridge | -0.000358 [-0.000475, -0.000219] | 100.0% |
| algo6_jinc3 | q10 | soft_chroma | +0.000072 [+0.000017, +0.000131] | 25.0% |
| algo6_jinc3 | q10_noise | coedge | -0.000603 [-0.000661, -0.000542] | 100.0% |
| algo6_jinc3 | q10_noise | isoluminant | -0.000565 [-0.000628, -0.000509] | 100.0% |
| algo6_jinc3 | q10_noise | luma_only | -0.001000 [-0.001141, -0.000857] | 100.0% |
| algo6_jinc3 | q10_noise | misaligned | -0.000587 [-0.000665, -0.000502] | 100.0% |
| algo6_jinc3 | q10_noise | ridge | -0.000518 [-0.000615, -0.000418] | 100.0% |
| algo6_jinc3 | q10_noise | soft_chroma | -0.000142 [-0.000186, -0.000085] | 87.5% |
| algo6_lanczos4 | clean | coedge | -0.000377 [-0.000435, -0.000316] | 100.0% |
| algo6_lanczos4 | clean | isoluminant | -0.000330 [-0.000411, -0.000262] | 100.0% |
| algo6_lanczos4 | clean | luma_only | +0.000000 [+0.000000, +0.000000] | 0.0% |
| algo6_lanczos4 | clean | misaligned | -0.000395 [-0.000470, -0.000315] | 100.0% |
| algo6_lanczos4 | clean | ridge | -0.000310 [-0.000408, -0.000193] | 100.0% |
| algo6_lanczos4 | clean | soft_chroma | +0.000026 [-0.000026, +0.000078] | 37.5% |
| algo6_lanczos4 | q10 | coedge | -0.000379 [-0.000437, -0.000317] | 100.0% |
| algo6_lanczos4 | q10 | isoluminant | -0.000333 [-0.000413, -0.000269] | 100.0% |
| algo6_lanczos4 | q10 | luma_only | +0.000000 [-0.000000, +0.000000] | 37.5% |
| algo6_lanczos4 | q10 | misaligned | -0.000399 [-0.000475, -0.000319] | 100.0% |
| algo6_lanczos4 | q10 | ridge | -0.000311 [-0.000407, -0.000196] | 100.0% |
| algo6_lanczos4 | q10 | soft_chroma | +0.000026 [-0.000024, +0.000075] | 37.5% |
| algo6_lanczos4 | q10_noise | coedge | -0.000595 [-0.000639, -0.000551] | 100.0% |
| algo6_lanczos4 | q10_noise | isoluminant | -0.000574 [-0.000619, -0.000534] | 100.0% |
| algo6_lanczos4 | q10_noise | luma_only | -0.001115 [-0.001276, -0.000955] | 100.0% |
| algo6_lanczos4 | q10_noise | misaligned | -0.000556 [-0.000598, -0.000513] | 100.0% |
| algo6_lanczos4 | q10_noise | ridge | -0.000487 [-0.000564, -0.000411] | 100.0% |
| algo6_lanczos4 | q10_noise | soft_chroma | -0.000210 [-0.000247, -0.000162] | 100.0% |

## TRecon Primary Results

| observation | motion px/frame | edge MAE off | edge MAE on | paired delta [95% CI] | scenes improved | full MAE delta | smooth MAE delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | 0.025131 | 0.024941 | -0.000190 [-0.000233, -0.000146] | 87.5% | -0.000029 | +0.000001 |
| clean | 1 | 0.025199 | 0.025062 | -0.000137 [-0.000177, -0.000097] | 81.2% | -0.000023 | +0.000001 |
| clean | 2 | 0.022002 | 0.021833 | -0.000169 [-0.000217, -0.000123] | 81.2% | -0.000031 | +0.000001 |
| clean | 3 | 0.022108 | 0.021978 | -0.000130 [-0.000170, -0.000089] | 81.2% | -0.000023 | +0.000001 |
| q10_noise | 0 | 0.027174 | 0.026529 | -0.000645 [-0.000710, -0.000584] | 100.0% | -0.001618 | -0.001783 |
| q10_noise | 1 | 0.027156 | 0.026580 | -0.000577 [-0.000655, -0.000503] | 100.0% | -0.001612 | -0.001782 |
| q10_noise | 2 | 0.024867 | 0.024225 | -0.000642 [-0.000717, -0.000574] | 100.0% | -0.001630 | -0.001778 |
| q10_noise | 3 | 0.024199 | 0.023592 | -0.000607 [-0.000691, -0.000534] | 100.0% | -0.001628 | -0.001791 |

## Largest Edge-MAE Regressions

- `bm_holdout_03051_straight_soft_chroma` (Recon, algo6_jinc3, obs=q10, D=box, siting=left, motion=): +0.000324
- `bm_holdout_03059_chevron_soft_chroma` (Recon, algo6_jinc3, obs=clean, D=lanczos, siting=left, motion=): +0.000308
- `bm_holdout_03059_chevron_soft_chroma` (Recon, algo6_jinc3, obs=q10, D=lanczos, siting=left, motion=): +0.000301
- `bm_holdout_03051_straight_soft_chroma` (Recon, algo6_jinc3, obs=clean, D=box, siting=left, motion=): +0.000297
- `bm_holdout_03059_chevron_soft_chroma` (Recon, algo6_jinc3, obs=clean, D=bicubic, siting=left, motion=): +0.000273

The `q10_noise` condition uses synthetic independent chroma noise and
is not a codec model. These results do not establish subjective benefit
or expected prevalence on released video.
