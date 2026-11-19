# SVHN (B1): Experiment 1 curriculum on SVHN

Curriculum class indices are the CIFAR ones; on SVHN these are digits. torchvision maps SVHN label 10 to digit 0.

| phase   | digits              |   n_samples |   steps_per_epoch |   epochs |   phase_start |   phase_end |   epoch0_start |   epoch0_end | epoch0_checks   |
|:--------|:--------------------|------------:|------------------:|---------:|--------------:|------------:|---------------:|-------------:|:----------------|
| Easy    | 0 1 8 9             |       28513 |               223 |        5 |             1 |        1115 |              1 |          223 | 150             |
| Medium  | 0 1 8 9 2 7         |       44693 |               350 |        5 |          1116 |        2865 |           1116 |         1465 | 1200 1350       |
| Hard    | 0 1 2 3 4 5 6 7 8 9 |       73257 |               573 |       10 |          2866 |        8595 |           2866 |         3438 | 3000 3150 3300  |

Instrumented runs reproduce the unchanged `run("exp1", datasets=("svhn",))` per-seed counts in **20/20** seeds.


## Per-seed (original code)

|   seed |   easy_to_med |   med_to_hard |     acc |
|-------:|--------------:|--------------:|--------:|
|     42 |            48 |            31 | 91.4413 |
|      7 |            34 |            27 | 90.9842 |
|    123 |            48 |            32 | 91.1071 |
|      0 |            43 |            10 | 91.4451 |
|     99 |            41 |            33 | 91.1955 |
|     11 |            32 |            34 | 91.1186 |
|     22 |            35 |            24 | 90.6154 |
|     33 |            34 |             4 | 90.8958 |
|     44 |            50 |             8 | 90.7498 |
|     55 |            40 |            15 | 91.1494 |
|    100 |            40 |            36 | 90.8305 |
|    101 |            45 |             9 | 91.4451 |
|    102 |            40 |             7 | 90.3734 |
|    103 |            44 |            16 | 91.3376 |
|    104 |            41 |            32 | 91.3337 |
|    105 |            48 |            20 | 91.1186 |
|    106 |            45 |            17 | 91.5834 |
|    107 |            49 |            23 | 90.7153 |
|    108 |            45 |            26 | 90.8075 |
|    109 |            49 |            25 | 91.1071 |

### SVHN statistics (n = 20 unless noted)

| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |
|---|---|---|---|---|---|---|---|---|---|---|
| SVHN original metric (epoch-0 deaths) | 4-class→6 | 6→10 | 42.55 ± 5.42 | 21.45 ± 9.91 | 0.50× [0.40, 0.61] | -7.910 | 1.98e-07 | 1.03e-04 | -1.77 | 1/0/19 (n=20) |
| SVHN original metric, batch 1 | 4-class→6 | 6→10 | 40.50 ± 6.30 | 21.80 ± 10.90 | 0.54× [0.36, 0.72] | -4.349 | 0.0019 | 0.0039 | -1.38 | 1/0/9 (n=10) |
| SVHN original metric, batch 2 | 4-class→6 | 6→10 | 44.60 ± 3.26 | 21.10 ± 8.79 | 0.47× [0.35, 0.61] | -7.332 | 4.41e-05 | 0.0020 | -2.32 | 0/0/10 (n=10) |
| SVHN: epoch-0 deaths | Medium | Hard | 42.55 ± 5.42 | 21.45 ± 9.91 | 0.50× [0.40, 0.61] | -7.910 | 1.98e-07 | 1.03e-04 | -1.77 | 1/0/19 (n=20) |
| SVHN: deaths per check, epoch 0 | Medium | Hard | 21.27 ± 2.71 | 7.15 ± 3.30 | 0.34× [0.27, 0.41] | -13.934 | 2.00e-11 | 8.82e-05 | -3.12 | 0/0/20 (n=20) |
| SVHN: first 150 steps | Medium | Hard | 3.30 ± 2.67 | 7.95 ± 4.63 | 2.41× [1.47, 4.29] | 3.244 | 0.0043 | 0.0103 | 0.73 | 12/1/7 (n=20) |
| SVHN: first 300 steps | Medium | Hard | 42.55 ± 5.42 | 18.25 ± 8.93 | 0.43× [0.34, 0.53] | -9.436 | 1.33e-08 | 1.03e-04 | -2.11 | 1/0/19 (n=20) |
| SVHN deaths at first full post-shift check (step 1350 vs 3150) | Medium | Hard | 39.25 ± 4.29 | 10.30 ± 5.93 | 0.26× [0.20, 0.33] | -17.359 | 4.11e-13 | 8.72e-05 | -3.88 | 0/0/20 (n=20) |
| SVHN eligible fraction at first full post-shift check | Medium | Hard | 0.64 ± 0.04 | 0.64 ± 0.08 | 1.00× [0.94, 1.07] | 0.008 | 0.9934 | 0.8983 | 0.00 | 9/0/11 (n=20) |

## Deaths at each check near the transitions (mean ± SD over seeds)

| phase   |   epoch |   step |   window_steps_in_phase |   deaths_mean |   deaths_sd |   alive_before |   eligible |
|:--------|--------:|-------:|------------------------:|--------------:|------------:|---------------:|-----------:|
| Hard    |       0 |   3000 |                     135 |          7.95 |        4.63 |          33.3  |       7.95 |
| Hard    |       0 |   3150 |                     150 |         10.3  |        5.93 |          26.3  |      17.1  |
| Hard    |       0 |   3300 |                     150 |          3.2  |        3.31 |          19.2  |       9.35 |
| Hard    |       1 |   3450 |                     150 |          3.5  |        3.23 |          19.5  |       9.25 |
| Medium  |       0 |   1200 |                      85 |          3.3  |        2.67 |          60.5  |       3.3  |
| Medium  |       0 |   1350 |                     150 |         39.25 |        4.29 |          61.2  |      39.35 |
| Medium  |       1 |   1500 |                     150 |          4.35 |        4.16 |          27.45 |       4.45 |
| Medium  |       1 |   1650 |                     150 |          4.85 |        3.68 |          29.1  |       4.9  |

Final 10-digit test accuracy: 91.07 ± 0.31%
