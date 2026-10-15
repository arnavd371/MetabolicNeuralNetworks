"""Analyse Parts 1-3.  python scripts/analyze.py results [figures]
-> results/analysis.md, results/analysis.json, figures/deaths_per_check_4panel.png.

Statistics battery (same definitions as metnn_exp1): mean ± SD with SD = np.std
(ddof=0, as in the original printouts), paired t, Wilcoxon signed-rank,
Cohen's d = mean(diff)/sd(diff, ddof=1), bootstrap 95% CI on the ratio of means
(10000 paired resamples, seed 0, +1e-6 in the denominator), consistency = #seeds with b > a.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats as st

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_MD, OUT = [], {}
B1 = [42, 7, 123, 0, 99, 11, 22, 33, 44, 55]


def boot_ratio(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = b[idx].mean() / (a[idx].mean() + 1e-6)
    lo, hi = np.percentile(out, [2.5, 97.5]); return float(lo), float(hi)


def battery(name, a, b, la, lb):
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a
    t, p = st.ttest_rel(b, a) if d.std() > 0 else (np.nan, np.nan)
    try: wp = st.wilcoxon(b, a).pvalue
    except ValueError: wp = np.nan
    cd = d.mean() / (d.std(ddof=1) + 1e-12)
    lo, hi = boot_ratio(a, b)
    res = dict(name=name, a=la, b=lb, n=len(a), a_mean=a.mean(), a_sd=a.std(),
               b_mean=b.mean(), b_sd=b.std(), ratio=b.mean() / (a.mean() + 1e-6),
               ci_lo=lo, ci_hi=hi, t=float(t), p=float(p), wilcoxon_p=float(wp), d=float(cd),
               b_gt_a=int((b > a).sum()), ties=int((b == a).sum()), b_lt_a=int((b < a).sum()))
    OUT[name] = res
    return res


def fmt_p(p):
    return "n/a" if np.isnan(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")


def table(rows, title):
    OUT_MD.append(f"\n### {title}\n")
    OUT_MD.append("| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |")
    OUT_MD.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        OUT_MD.append(f"| {r['name']} | {r['a']} | {r['b']} | {r['a_mean']:.2f} ± {r['a_sd']:.2f} | "
                      f"{r['b_mean']:.2f} ± {r['b_sd']:.2f} | {r['ratio']:.2f}× [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
                      f"{r['t']:.3f} | {fmt_p(r['p'])} | {fmt_p(r['wilcoxon_p'])} | {r['d']:.2f} | "
                      f"{r['b_gt_a']}/{r['ties']}/{r['b_lt_a']} (n={r['n']}) |")


def window(ck, phase, lo, hi):
    """Sum of deaths over checks with lo <= step_in_phase <= hi, per seed."""
    m = ck[(ck.phase == phase) & (ck.step_in_phase >= lo) & (ck.step_in_phase <= hi)]
    return m.groupby("seed").deaths.sum()


def per_epoch(ep, phase, e, col="deaths"):
    m = ep[(ep.phase == phase) & (ep.epoch == e)].set_index("seed")
    return m[col], m.n_checks


def phase_len(ep, phase):
    m = ep[ep.phase == phase]; return int(m.n_steps.sum() / m.seed.nunique())


def curriculum_block(ck, ep, cur, A, B, label):
    """Standard battery for one curriculum: transition into phase A vs into phase B."""
    c, e = ck[ck.curriculum == cur], ep[ep.curriculum == cur]
    seeds = sorted(e.seed.unique())
    dA, nA = per_epoch(e, A, 0); dB, nB = per_epoch(e, B, 0)
    rows = [battery(f"{label}: epoch-0 deaths", dA[seeds], dB[seeds], A, B)]
    OUT[f"{label}: checks in epoch 0"] = {A: sorted(set(nA)), B: sorted(set(nB))}
    rows.append(battery(f"{label}: deaths per check, epoch 0", dA[seeds] / nA[seeds],
                        dB[seeds] / nB[seeds], A, B))
    for w in (150, 300):
        rows.append(battery(f"{label}: first {w} steps", window(c, A, 1, w).reindex(seeds, fill_value=0),
                            window(c, B, 1, w).reindex(seeds, fill_value=0), A, B))
    return rows


def main():
    figs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(R)), "figures")
    os.makedirs(figs, exist_ok=True)
    # ---------------- Part 1 ----------------
    p1 = pd.read_csv(f"{R}/part1_per_seed.csv")
    repo = {42: (7, 18), 7: (2, 28), 123: (4, 34), 0: (0, 31), 99: (8, 24), 11: (9, 18), 22: (15, 20),
            33: (6, 34), 44: (8, 39), 55: (1, 22), 100: (12, 24), 101: (0, 29), 102: (4, 25), 103: (2, 23),
            104: (0, 46), 105: (0, 33), 106: (13, 25), 107: (0, 35), 108: (7, 32), 109: (4, 16)}
    p1["repo_easy"] = p1.seed.map(lambda s: repo[s][0]); p1["repo_hard"] = p1.seed.map(lambda s: repo[s][1])
    p1["diff_easy"] = p1.easy_to_med - p1.repo_easy; p1["diff_hard"] = p1.med_to_hard - p1.repo_hard
    p1.to_csv(f"{R}/part1_vs_repo.csv", index=False)
    OUT["part1_exact_seed_matches"] = int(((p1.diff_easy == 0) & (p1.diff_hard == 0)).sum())
    OUT_MD.append("## Part 1: per-seed reproduction vs repo\n")
    OUT_MD.append(p1.to_markdown(index=False))
    rows = [battery("Part1 batch1", p1[p1.batch == "batch1"].easy_to_med, p1[p1.batch == "batch1"].med_to_hard, "E→M", "M→H"),
            battery("Part1 batch2", p1[p1.batch == "batch2"].easy_to_med, p1[p1.batch == "batch2"].med_to_hard, "E→M", "M→H"),
            battery("Part1 pooled", p1.easy_to_med, p1.med_to_hard, "E→M", "M→H")]
    table(rows, "Part 1 statistics (my reproduction)")

    # ---------------- Part 2 ----------------
    ck = pd.read_csv(f"{R}/part2_per_check.csv"); ep = pd.read_csv(f"{R}/part2_per_epoch.csv")
    ps = pd.read_csv(f"{R}/part2_per_seed.csv")
    orig = ps[ps.curriculum == "original"].set_index("seed")
    m = p1.set_index("seed")
    common = sorted(set(orig.index) & set(m.index))
    same = ((orig.loc[common, "td_Medium"] == m.loc[common, "easy_to_med"]) &
            (orig.loc[common, "td_Hard"] == m.loc[common, "med_to_hard"]))
    OUT["instrumented_matches_part1"] = f"{int(same.sum())}/{len(common)}"

    # verification of the check schedule
    sched = (ck[ck.curriculum == "original"].groupby(["phase", "epoch"])
             .agg(checks_per_seed=("step", lambda s: len(s) / ck.seed.nunique()),
                  check_steps=("step", lambda s: sorted(set(s))[:4])).reset_index())
    e0 = ep[(ep.curriculum == "original") & (ep.epoch <= 1)][["phase", "epoch", "start_step", "end_step", "n_steps", "n_checks"]].drop_duplicates()
    OUT_MD.append("\n## Part 2: check schedule (logged, original curriculum)\n")
    OUT_MD.append(e0.to_markdown(index=False))
    OUT_MD.append("\nCheck steps (first 4 per phase/epoch):\n")
    OUT_MD.append(sched[sched.epoch <= 1].to_markdown(index=False))

    co = ck[ck.curriculum == "original"]; eo = ep[ep.curriculum == "original"]; seeds = sorted(eo.seed.unique())
    rows = curriculum_block(ck, ep, "original", "Medium", "Hard", "orig")
    # individual checks
    def check_k(phase, k):
        mm = co[(co.phase == phase) & (co.epoch == 0)].sort_values("step").groupby("seed").nth(k)
        return mm.set_index("seed").deaths.reindex(seeds)
    rows.append(battery("orig: Medium check#1 vs Hard check#1 (epoch 0)", check_k("Medium", 0), check_k("Hard", 0), "Medium c1", "Hard c1"))
    rows.append(battery("orig: Medium check#1 vs Hard check#2 (epoch 0)", check_k("Medium", 0), check_k("Hard", 1), "Medium c1", "Hard c2"))
    table(rows, "Part 2a/2b: original curriculum: raw, per-check and fixed-window")

    # (c) no-shift baselines within the same runs
    rows = []
    for ph in ("Medium", "Hard"):
        d0, n0 = per_epoch(eo, ph, 0); d1, n1 = per_epoch(eo, ph, 1)
        rows.append(battery(f"{ph}: per-check, epoch 1 (A) vs epoch 0 (B)", d1[seeds] / n1[seeds], d0[seeds] / n0[seeds], f"{ph} ep1", f"{ph} ep0"))
        L = phase_len(eo, ph); mid = L // 2
        rows.append(battery(f"{ph}: mid-phase 300 (A) vs first 300 (B)", window(co, ph, mid + 1, mid + 300).reindex(seeds, fill_value=0),
                            window(co, ph, 1, 300).reindex(seeds, fill_value=0), f"{ph} mid300", f"{ph} first300"))
    # transition excess over the no-shift baseline: compare excess between transitions
    def excess(ph):
        L = phase_len(eo, ph); mid = L // 2
        return (window(co, ph, 1, 300) - window(co, ph, mid + 1, mid + 300)).reindex(seeds, fill_value=0)
    rows.append(battery("excess deaths (first300 − mid300): Medium (A) vs Hard (B)", excess("Medium"), excess("Hard"), "Med excess", "Hard excess"))
    table(rows, "Part 2c: no-shift baselines in the same runs")

    # (d) control curricula
    ctrl = []
    if "noshift_matched" in set(ps.curriculum):
        ctrl += curriculum_block(ck, ep, "noshift_matched", "Medium", "Hard", "noshift_matched (4→4→4, original timing)")
        ns = ep[(ep.curriculum == "noshift_matched") & (ep.epoch == 0)]
        a = ns[ns.phase == "Hard"].set_index("seed").deaths.reindex(seeds)
        b = eo[(eo.phase == "Hard") & (eo.epoch == 0)].set_index("seed").deaths.reindex(seeds)
        ctrl.append(battery("Hard-slot epoch-0 deaths: no shift (A) vs real 6→10 shift (B)", a, b, "no shift", "6→10"))
        a = ns[ns.phase == "Medium"].set_index("seed").deaths.reindex(seeds)
        b = eo[(eo.phase == "Medium") & (eo.epoch == 0)].set_index("seed").deaths.reindex(seeds)
        ctrl.append(battery("Medium-slot epoch-0 deaths: no shift (A) vs real 4→6 shift (B)", a, b, "no shift", "4→6"))
        cn = ck[ck.curriculum == "noshift_matched"]
        for w in (150, 300):
            ex = {ph: (window(co, ph, 1, w) - window(cn, ph, 1, w)).reindex(seeds, fill_value=0) for ph in ("Medium", "Hard")}
            ctrl.append(battery(f"excess over matched no-shift run, first {w} steps: 4→6 (A) vs 6→10 (B)",
                                ex["Medium"], ex["Hard"], "4→6 excess", "6→10 excess"))
    # deaths normalised by the alive population at the first check of each phase
    for w in (150, 300):
        fr = {}
        for ph in ("Medium", "Hard"):
            m = co[(co.phase == ph) & (co.step_in_phase <= w)]
            first_alive = m.sort_values("step").groupby("seed").alive_before.first()
            fr[ph] = (m.groupby("seed").deaths.sum() / first_alive).reindex(seeds)
        ctrl.append(battery(f"orig: deaths / alive at phase start, first {w} steps", fr["Medium"], fr["Hard"], "Medium", "Hard"))
    # first check whose 150-step rate window lies entirely after the shift:
    # eligible fraction (r < theta_death among alive, before the N_min floor)
    def full_window_frac(c, ph):
        m = c[(c.phase == ph) & (c.window_steps_in_phase == 150)].sort_values("step").groupby("seed").first()
        return (m.below_death_before / m.alive_before).reindex(seeds), int(m.step.iloc[0])
    fm, sm = full_window_frac(co, "Medium"); fh, sh = full_window_frac(co, "Hard")
    ctrl.append(battery(f"orig: eligible fraction at first full post-shift check (step {sm} vs {sh})", fm, fh, "Medium", "Hard"))
    if "noshift_matched" in set(ps.curriculum):
        nm, _ = full_window_frac(cn, "Medium"); nh, _ = full_window_frac(cn, "Hard")
        ctrl.append(battery("same, no-shift matched run (timing control)", nm, nh, "Medium slot", "Hard slot"))
        ctrl.append(battery("same, excess over no-shift run", fm - nm, fh - nh, "4→6", "6→10"))
    OUT["floor_binding_checks"] = int((co.below_death_before > co.deaths).sum())
    OUT["alive_before_first_check"] = co[co.step.isin([900, 2100])].groupby("step").alive_before.agg(["mean", "std"]).round(1).to_dict()
    if "equal_len" in set(ps.curriculum):
        ctrl += curriculum_block(ck, ep, "equal_len", "Medium", "Hard", "equal_len (4→6→10, 157 steps/epoch)")
    if "reversed" in set(ps.curriculum):
        ctrl += curriculum_block(ck, ep, "reversed", "Medium", "Easy", "reversed (10→6→4)")
    table(ctrl, "Part 2d: control curricula (A = first transition, B = second transition)")
    OUT_MD.append("\nChecks in epoch 0 per curriculum: " + json.dumps({k: v for k, v in OUT.items() if "checks in epoch 0" in k}, default=str))

    # accuracy by curriculum
    OUT_MD.append("\n### Final test accuracy by curriculum\n")
    OUT_MD.append(ps.groupby("curriculum").acc.agg(["mean", "std", "count"]).round(2).to_markdown())

    # figure: mean deaths per check vs global step, original vs no-shift-matched
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, cur in zip(axes.flat, ["original", "noshift_matched", "equal_len", "reversed"]):
        c = ck[ck.curriculum == cur]
        if c.empty: ax.set_visible(False); continue
        g = c.groupby("step").deaths.agg(["mean", "sem"]).reset_index()
        ax.bar(g.step, g["mean"], width=110, yerr=g["sem"], color="#4C72B0", ecolor="#999", capsize=1.5)
        for _, rr in ep[(ep.curriculum == cur) & (ep.epoch == 0)].drop_duplicates("phase").iterrows():
            ax.axvline(rr.start_step - 0.5, color="#C44E52", ls="--", lw=1)
            ax.axvspan(rr.start_step, rr.end_step, color="#C44E52", alpha=0.08)
            ax.text(rr.start_step + 20, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 1 else 1, rr.phase, fontsize=8, color="#C44E52")
        ax.set_title(cur, fontsize=10); ax.set_xlabel("global step"); ax.set_ylabel("deaths per check (mean ± SEM)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Deaths at each reorganize() check (shaded = epoch 0 of each phase, the window the paper counts)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{figs}/deaths_per_check_4panel.png", dpi=150); plt.close(fig)

    # ---------------- Part 3 ----------------
    if os.path.exists(f"{R}/part3_per_check.csv"):
        ck3 = pd.read_csv(f"{R}/part3_per_check.csv"); ep3 = pd.read_csv(f"{R}/part3_per_epoch.csv")
        ps3 = pd.read_csv(f"{R}/part3_per_seed.csv")
        rows = curriculum_block(ck3, ep3, "original", "Medium", "Hard", "binary rate")
        rows.append(battery("frac rate (Part 2 runs, same 10 seeds): epoch-0 deaths",
                            orig.loc[B1, "td_Medium"], orig.loc[B1, "td_Hard"], "Medium", "Hard"))
        table(rows, "Part 3: binary per-pass activation indicator (original curriculum)")
        OUT_MD.append("\nBinary-rate runs: total deaths per seed over the whole run: " +
                      str(ck3.groupby("seed").deaths.sum().to_dict()) +
                      f"; mean final accuracy {ps3.acc.mean():.2f}%; alive at end {ps3['alive_end_Hard'].tolist()}")
        OUT_MD.append("\nMean activation rate of alive neurons at checks (binary): "
                      f"{ck3.mean_r_alive.mean():.3f} (min {ck3.mean_r_alive.min():.3f}); "
                      f"alive neurons with r<0.05 before a check, max over all checks: {ck3.below_death_before.max()}")

    open(f"{R}/analysis.md", "w").write("\n".join(OUT_MD) + "\n")
    json.dump(OUT, open(f"{R}/analysis.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n".join(OUT_MD))
    print("\ninstrumented == part1:", OUT["instrumented_matches_part1"], " exact repo matches:", OUT["part1_exact_seed_matches"], "/ 20")


if __name__ == "__main__":
    main()
