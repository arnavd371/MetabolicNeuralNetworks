"""B1/B2 analysis.  python analyze_svhn_fisher.py <results_dir>
-> svhn_analysis.md, fisher_analysis.md (same statistics battery as analyze.py)."""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats as st

sys.argv = sys.argv[:2]
import analyze as A                       # battery(), table(), window(), per_epoch()

R = sys.argv[1] if len(sys.argv) > 1 else "results"
B1 = [42, 7, 123, 0, 99, 11, 22, 33, 44, 55]


def svhn():
    A.OUT_MD.clear()
    sch = pd.read_csv(f"{R}/svhn_schedule.csv")
    ck = pd.read_csv(f"{R}/svhn_per_check.csv"); ep = pd.read_csv(f"{R}/svhn_per_epoch.csv")
    ps = pd.read_csv(f"{R}/svhn_per_seed.csv").set_index("seed")
    og = pd.read_csv(f"{R}/svhn_original_per_seed.csv").set_index("seed")
    seeds = sorted(ps.index)
    md = A.OUT_MD
    md.append("# SVHN (B1): Experiment 1 curriculum on SVHN\n")
    md.append("Curriculum class indices are the CIFAR ones; on SVHN these are digits. "
              "torchvision maps SVHN label 10 to digit 0.\n")
    md.append(sch.to_markdown(index=False))
    match = ((ps.loc[seeds, "td_Medium"] == og.loc[seeds, "easy_to_med"]) &
             (ps.loc[seeds, "td_Hard"] == og.loc[seeds, "med_to_hard"]))
    md.append(f"\nInstrumented runs reproduce the unchanged `run(\"exp1\", datasets=(\"svhn\",))` "
              f"per-seed counts in **{int(match.sum())}/{len(seeds)}** seeds.\n")
    md.append("\n## Per-seed (original code)\n")
    md.append(og.reset_index().to_markdown(index=False))

    rows = [A.battery("SVHN original metric (epoch-0 deaths)", og.loc[seeds, "easy_to_med"],
                      og.loc[seeds, "med_to_hard"], "4-class→6", "6→10")]
    for name, ss in (("batch 1", B1), ("batch 2", [s for s in seeds if s not in B1])):
        rows.append(A.battery(f"SVHN original metric, {name}", og.loc[ss, "easy_to_med"],
                              og.loc[ss, "med_to_hard"], "4-class→6", "6→10"))
    ck["curriculum"] = "original"; ep["curriculum"] = "original"
    rows += A.curriculum_block(ck, ep, "original", "Medium", "Hard", "SVHN")

    def full(ph):
        m = ck[(ck.phase == ph) & (ck.window_steps_in_phase == 150)].sort_values("step").groupby("seed").first()
        return m.reindex(seeds)
    fm, fh = full("Medium"), full("Hard")
    rows.append(A.battery(f"SVHN deaths at first full post-shift check (step {int(fm.step.iloc[0])} vs {int(fh.step.iloc[0])})",
                          fm.deaths, fh.deaths, "Medium", "Hard"))
    rows.append(A.battery("SVHN eligible fraction at first full post-shift check",
                          fm.below_death_before / fm.alive_before, fh.below_death_before / fh.alive_before, "Medium", "Hard"))
    A.table(rows, "SVHN statistics (n = 20 unless noted)")

    md.append("\n## Deaths at each check near the transitions (mean ± SD over seeds)\n")
    near = ck[(ck.phase != "Easy") & (ck.step_in_phase <= 600)]
    t = near.groupby(["phase", "epoch", "step", "window_steps_in_phase"]).agg(
        deaths_mean=("deaths", "mean"), deaths_sd=("deaths", lambda x: x.std(ddof=0)),
        alive_before=("alive_before", "mean"), eligible=("below_death_before", "mean")).round(2).reset_index()
    md.append(t.to_markdown(index=False))
    md.append(f"\nFinal 10-digit test accuracy: {og.acc.mean():.2f} ± {og.acc.std(ddof=0):.2f}%")
    open(f"{R}/svhn_analysis.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def fisher():
    f = pd.read_csv(f"{R}/fisher_vs_rate.csv")
    rho = f.spearman_rho.values
    t, p = st.ttest_1samp(rho, 0.0)
    npos = int((rho > 0).sum())
    sign_p = st.binomtest(npos, len(rho), 0.5).pvalue
    nsig = int(((f.p < 0.05) & (f.spearman_rho > 0)).sum()), int(((f.p < 0.05) & (f.spearman_rho < 0)).sum())
    md = ["# Fisher importance vs activation rate (B2)\n",
          "Setting: Experiment 1 model and curriculum, measured at the end of phase 1 (Easy, step 785). "
          "r_i = layer activation rate (last 150 steps); F_i = Experiment 2 `_fisher` (eval mode, fp32, "
          "mean of squared batch-loss gradients over 8 batches), summed per neuron over incoming weights "
          "and bias. Spearman correlation across alive neurons, per seed.\n",
          f.round(4).to_markdown(index=False),
          f"\nAcross {len(rho)} seeds: mean rho = {rho.mean():+.3f}, SD = {rho.std(ddof=1):.3f}, "
          f"median = {np.median(rho):+.3f}, range [{rho.min():+.3f}, {rho.max():+.3f}].",
          f"One-sample t-test of rho against 0: t = {t:.3f}, p = {p:.3g}.",
          f"Sign test: {npos}/{len(rho)} positive, p = {sign_p:.3g}.",
          f"Per-seed significance at p < 0.05: {nsig[0]} positive, {nsig[1]} negative.",
          f"Alive neurons per seed: {int(f.n_alive.min())}–{int(f.n_alive.max())} (mean {f.n_alive.mean():.1f})."]
    open(f"{R}/fisher_analysis.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if os.path.exists(f"{R}/fisher_vs_rate.csv"): fisher()
    if os.path.exists(f"{R}/svhn_per_check.csv"): svhn()
