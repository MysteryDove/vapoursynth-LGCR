# LGCR gate evaluation

```text
[ 25/156] edge_s0_m4_a0_c0.55
[ 50/156] edge_s2_m2_a0_c0.2
[ 75/156] edge_s3_m1_a22_c0.55
[100/156] edge_s5_m0.5_a22_c0.2
[125/156] edge_s7_m0_a45_c0.55
[150/156] edge_s7_m4_a45_c0.2
[156/156] ramp

156 unique samples x 3 actual degradations = 468 observations

--- label: ungated algo=2 mechanism vs plain ---
468 observations, 420 labeled (135 benefit / 285 harm), 48 neutral excluded; mean delta=+0.00263
statistic       AUC   risk at target coverage (target->actual:risk; lower is better)
cedge_proxy   0.632     5%->13.6%:-0.0003   10%->13.6%:-0.0003   20%->20.0%:+0.0003   50%->50.0%:+0.0019  100%->100.0%:+0.0029
width_eq      0.664     5%-> 5.0%:-0.0015   10%->10.0%:+0.0007   20%->20.0%:+0.0000   50%->50.0%:+0.0010  100%->100.0%:+0.0029
ms            0.711     5%->29.5%:-0.0016   10%->29.5%:-0.0016   20%->29.5%:-0.0016   50%->50.0%:+0.0017  100%->100.0%:+0.0029
per-actual-D AUC:
statistic          box       tri       bic
cedge_proxy      0.505     0.681     0.713
width_eq         0.445     0.815     0.833
ms               0.865     0.752     0.681

--- label: q/ms-independent algo=6 detail transfer vs plain ---
468 observations, 441 labeled (205 benefit / 236 harm), 27 neutral excluded; mean delta=+0.00003
statistic       AUC   risk at target coverage (target->actual:risk; lower is better)
q_box         0.745     5%-> 5.2%:-0.0054   10%->10.2%:-0.0048   20%->20.4%:-0.0045   50%->50.1%:-0.0020  100%->100.0%:+0.0000
q_min         0.732     5%-> 5.2%:-0.0062   10%->10.2%:-0.0052   20%->20.2%:-0.0032   50%->50.1%:-0.0014  100%->100.0%:+0.0000
q_prod        0.724     5%-> 5.2%:-0.0066   10%->10.2%:-0.0053   20%->20.2%:-0.0030   50%->50.1%:-0.0013  100%->100.0%:+0.0000
ms            0.769     5%->28.8%:-0.0052   10%->28.8%:-0.0052   20%->28.8%:-0.0052   50%->50.1%:-0.0024  100%->100.0%:+0.0000
q_prod_ms     0.777     5%-> 5.2%:-0.0066   10%->10.2%:-0.0053   20%->20.2%:-0.0054   50%->50.1%:-0.0025  100%->100.0%:+0.0000
per-actual-D AUC:
statistic          box       tri       bic
q_box            0.992     0.959     0.604
q_min            0.992     0.962     0.632
q_prod           0.992     0.946     0.622
ms               0.966     0.974     0.646
q_prod_ms        0.982     0.990     0.660

full algo=6 mean delta vs plain by actual D:
  box  -0.00189
  tri  -0.00150
  bic  -0.00026
  all  -0.00122
full algo=6 labeled outcomes: 158 benefit / 88 harm / 222 neutral
worst full algo=6 residuals:
  edge_s5_m0.5_a45_c0.55     D=bic delta=+0.00844 q*ms=0.676
  edge_s5_m0_a45_c0.55       D=bic delta=+0.00765 q*ms=0.675
  edge_s5_m0_a22_c0.55       D=bic delta=+0.00737 q*ms=0.577
  edge_s5_m0.5_a22_c0.55     D=bic delta=+0.00648 q*ms=0.580
  edge_s3_m1_a0_c0.55        D=bic delta=+0.00514 q*ms=0.714
  edge_s5_m0.5_a0_c0.55      D=bic delta=+0.00459 q*ms=0.399
  edge_s5_m1_a45_c0.55       D=bic delta=+0.00454 q*ms=0.664
  edge_s5_m0_a45_c0.55       D=tri delta=+0.00444 q*ms=0.529
```
