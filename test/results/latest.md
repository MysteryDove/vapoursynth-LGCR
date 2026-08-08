# LGCR battery results

```
case              plain     algo2
---------------------------------
hard_v          0.03630   0.02361
hard_d45        0.08810   0.07974
isoluminant     0.03630   0.03630
misalign4       0.02030   0.02030
hardL_softC     0.00564   0.00592
ridge_line      0.00181   0.00181
ramp            0.00505   0.00505
texture         0.01196   0.01196
noise           0.03752   0.03758
upscale2x       0.03860   0.03367
temporal        0.00000   0.00000   (inter-frame diff in static region; lower=better)

temporal cases (plain / algo2 / trecon):
t_move1         0.03038   0.01587   0.01598
t_move2         0.03038   0.01587   0.01587
t_move3         0.03038   0.01587   0.01598
t_static_noise   0.03227   0.02846   0.02631
t_single      isolation maxdiff = 0.000000  (must be ~0: single frame has no temporal information)
t_flat_amb      0.01220   0.01220   (identical frames, unobservable motion: trecon ~= algo2 required)

int8 constant (64,128,128): {'recon8': [64, 128, 128], 'trecon8': [64, 128, 128]}  OK
```
