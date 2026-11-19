# Metabolic Neural Networks

Code, results and a full reproduction for **Metabolic Neural Networks (MetNN)**: a layer whose neurons are removed (and, in the full design, duplicated or frozen) based on one local statistic, each neuron's rolling activation rate, with no task labels, global criterion or fixed schedule.

## The mechanism

Each neuron in the metabolic layer tracks `r(i)`: the fraction of samples on which its pre-activation was positive, averaged over the last 150 training steps. Every 150 global steps:

| Decision | Condition | Effect |
|---|---|---|
| **Death** | `r(i) < 0.05` | Neuron removed (never below `N_min = 16` alive) |
| **Birth** | `r(i) > 0.70` during a loss plateau | Neuron duplicated, with noise |
| **Consolidation** | `r(i) > 0.30` after a burst of 5+ deaths | Neuron frozen (disabled in all experiments here) |

**Relation to prior work.** The death rule is a form of inactivity-based pruning, closely related to Network Trimming (arXiv:1607.03250) and Demon Pruning (arXiv:2403.07688). It is not a new pruning criterion. The contribution here is:
1. an exactness statement: under ReLU, a neuron with `r(i) = 0` contributed exactly zero gradient over its window, so removing it changes nothing already computed;
2. an analysis of the bursts of removals that follow a distribution shift;
3. a measurement protocol for those bursts.

## Experiment 1: pruning bursts after a curriculum shift

A 3-phase CIFAR-10 curriculum (4 classes, then 6, then all 10) with no signal about phase transitions. The **original metric** counts neurons removed in the first epoch after each transition, over two batches of 10 seeds.

**Original metric** (± = population SD over seeds):

| Seeds | 4→6 deaths | 6→10 deaths | Ratio | Paired t p | Cohen's d |
|---|---|---|---|---|---|
| Batch 1: 42, 7, 123, 0, 99, 11, 22, 33, 44, 55 | 6.0 ± 4.2 | 26.8 ± 7.1 | 4.47× | 9.3e-5 | 2.104 |
| Batch 2: 100–109 | 4.2 ± 4.7 | 28.8 ± 7.8 | 6.86× | 7.4e-5 | 2.168 |
| Pooled (n = 20) | 5.10 ± 4.57 | 27.80 ± 7.53 | 5.45× | 9.6e-9 | 2.154 |

**What the reproduction shows** (details in [REPRODUCTION.md](REPRODUCTION.md)):

- **The result reproduces exactly.** All 20/20 seeds give identical death counts on a Tesla T4 with PyTorch 2.11.0.
- **The 5.45× ratio is mostly a counting-window artifact, not a measure of shift severity.**
  - Reorganisation checks fire every 150 global steps. The first epoch after 4→6 contains 1 check; the first epoch after 6→10 contains 2.
  - A burst only appears once a neuron's 150-step rate window is post-shift. The 4→6 burst (step 1050) falls just after its counted epoch; the 6→10 burst (step 2250) falls inside its counted epoch.
  - A timing-matched run with **no distribution shift** already gives 2.49× on the same metric.
  - With equal windows after each transition, the ratio is 1.41× over 150 steps (p = 0.24) and 0.66× over 300 steps.
- **What survives:**
  - Class-adding shifts produce bursts above a no-shift baseline.
  - The larger shift silences a slightly larger fraction of the surviving neurons: 1.16×, 95% CI [1.08, 1.23], 16/20 seeds.
  - Absolute death counts do not track severity, and removing classes produces no burst.

## Experiment 2: continual learning (not reproduced)

`Experiment2.ipynb` holds the code. Despite the extension, the file is plain Python source, not notebook JSON: paste it into a notebook cell to run it. It compares a static baseline, EWC, and MetNN with and without consolidation on Split-CIFAR-10. **This experiment was not part of the reproduction pass.** The table below is as originally reported and is unverified.

| Method | Final accuracy | Forgetting |
|---|---|---|
| Static baseline | 76.25% | 16.83% |
| EWC | 83.13% | 7.07% |
| Metabolic, no protection | 74.59% | 18.89% |
| Metabolic, with protection | 76.16% | 16.13% |

## Usage (original notebooks)

Paste the code cell of `Experiment1.ipynb` into a notebook (or run `scripts/metnn_exp1.py`, the same code verbatim), then:

```python
run("smoke")                                                              # sanity check, synthetic data
run("exp1", seeds=[42,7,123,0,99,11,22,33,44,55], batch_name="batch1")
run("exp1", seeds=[100,101,102,103,104,105,106,107,108,109], batch_name="batch2")
run("meta"); run("figures"); run("summary")
```

## Reproduction

`scripts/` reproduces Experiment 1 and runs the controls. Each full training run takes about 30 s on a T4.

| Script | What it does |
|---|---|
| `metnn_exp1.py` | `Experiment1.ipynb` code cell, verbatim |
| `metnn_instr.py` | Instrumented copy of `_run_curriculum`: read-only per-check logging, no extra RNG draws; control curricula (`noshift_matched`, `equal_len`, `reversed`); binary-rate variant |
| `run_all.py` | Driver: `smoke`, `part1` (faithful, 20 seeds), `part2` (instrumented + controls, 80 runs), `part3` (binary rate, 10 seeds), `svhn` |
| `analyze.py`, `analyze_svhn_fisher.py` | Regenerate `results/analysis.md`, `svhn_analysis.md`, `fisher_analysis.md` from the CSV logs |
| `fisher_vs_rate.py` | Fisher importance vs activation rate |
| `make_figures.py` | Paper figures (`deaths_per_check_timeline`, `alive_vs_step`, `perseed_scatter`) |
| `build_kernels.py` | Packages runs as self-contained Kaggle GPU kernels (how these results were produced) |

```bash
pip install -r requirements.txt
cd scripts && python run_all.py part1 --out ../results     # needs a CUDA GPU; CPU works but is slow
cd .. && python scripts/analyze.py results                   # CPU, seconds
```

**Expected output to compare against:**
- `results/part1.txt`, for example seed 42: 7 deaths after 4→6, 18 after 6→10, 74.08% accuracy;
- the pooled statistics in the table above;
- `results/analysis.md` for every control.

**Hardware note.** Seed-level agreement was exact on a Kaggle Tesla T4 (driver 580.178, PyTorch 2.11.0+cu128, cuDNN 9.19, Python 3.13.15) with `cudnn.deterministic=True`. Other GPUs or drivers may differ at the seed level while agreeing statistically.

## Paper

*Metabolic Neural Networks: Exact Local Pruning from Neuron Activation Rates and the Limits of Burst-Based Shift Detection* (Arnav Dhiman, 2026). The LaTeX source is not yet in this repository; it will be added under `paper/`.

## Known limitations

- **One dataset in the paper.** All of the paper's experiments use CIFAR-10.
- **Birth rule not ablated.** It was active in every run, so all death counts come from a death-plus-birth system.
- **Severity is only class count.** Shift severity is measured solely by the number of classes added.
- **No baselines.** There is no comparison with a baseline shift detector (for example, loss-based) and no accuracy comparison with fixed-width networks.
- **SVHN and the Fisher-vs-rate correlation were removed from the paper.** They were later rerun on the same setup, and the reruns contradicted the earlier unreleased claims (see the addendum in [REPRODUCTION.md](REPRODUCTION.md)):
  - SVHN reverses on the original metric (0.50×, p = 2.0e-7).
  - The Fisher-vs-rate correlation is rho = +0.271 (one-sample p = 0.00246, 10 seeds), not 0.17 and non-significant.

## License

Code is released under the [MIT License](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

```bibtex
@article{dhiman2026metabolic,
  title  = {Metabolic Neural Networks: Exact Local Pruning from Neuron Activation Rates and the Limits of Burst-Based Shift Detection},
  author = {Dhiman, Arnav},
  year   = {2026},
  note   = {TODO: arXiv identifier},
  url    = {TODO}
}
```
