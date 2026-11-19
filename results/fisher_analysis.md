# Fisher importance vs activation rate (B2)

Setting: Experiment 1 model and curriculum, measured at the end of phase 1 (Easy, step 785). r_i = layer activation rate (last 150 steps); F_i = Experiment 2 `_fisher` (eval mode, fp32, mean of squared batch-loss gradients over 8 batches), summed per neuron over incoming weights and bias. Spearman correlation across alive neurons, per seed.

|   seed |   step |   n_alive |   spearman_rho |      p |   r_mean |   fisher_median |
|-------:|-------:|----------:|---------------:|-------:|---------:|----------------:|
|     42 |    785 |        78 |         0.433  | 0.0001 |   0.6902 |          0.0372 |
|      7 |    785 |        67 |         0.3489 | 0.0038 |   0.7368 |          0.0098 |
|    123 |    785 |        72 |         0.2242 | 0.0583 |   0.7865 |          0.0161 |
|      0 |    785 |        56 |        -0.2324 | 0.0848 |   0.8596 |          0.0161 |
|     99 |    785 |        88 |         0.4477 | 0      |   0.7282 |          0.0062 |
|     11 |    785 |        71 |         0.4217 | 0.0002 |   0.6811 |          0.007  |
|     22 |    785 |        77 |         0.3796 | 0.0007 |   0.6656 |          0.0029 |
|     33 |    785 |        87 |         0.2622 | 0.0141 |   0.7139 |          0.0058 |
|     44 |    785 |        67 |         0.3146 | 0.0095 |   0.7311 |          0.0145 |
|     55 |    785 |        77 |         0.1082 | 0.349  |   0.7993 |          0.0091 |

Across 10 seeds: mean rho = +0.271, SD = 0.206, median = +0.332, range [-0.232, +0.448].
One-sample t-test of rho against 0: t = 4.156, p = 0.00246.
Sign test: 9/10 positive, p = 0.0215.
Per-seed significance at p < 0.05: 7 positive, 0 negative.
Alive neurons per seed: 56–88 (mean 74.0).
