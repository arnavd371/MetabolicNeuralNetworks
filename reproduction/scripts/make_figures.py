"""Paper figures from the Part 2 logs.  python make_figures.py [results_dir] [figures_dir]

deaths_per_check_timeline.pdf, alive_vs_step.pdf, perseed_scatter.pdf (+ .png previews).
Original curriculum, 20 seeds. Bands = +/-1 population SD (ddof=0).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as blend

R = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(FIG, exist_ok=True)

# xcolor values used by the paper's pgfplots figures: blue!60!black (markers/lines),
# blue!25 (fills), orange!35 (no-shift fills; not needed in these three figures)
BLUE, BLUE_FILL, ORANGE, GRAY = "#000099", "#BFBFFF", "#FFC499", "#7f7f7f"
B1 = [42, 7, 123, 0, 99, 11, 22, 33, 44, 55]
BOUNDS = (785, 1960)                                   # last step of Easy / Medium
COUNTED = ((786, 1020), (1961, 2351))                  # epoch 0 of Medium / Hard
PHASES = (("Easy\n4 classes", 0, 785), ("Medium\n6 classes", 785, 1960), ("Hard\n10 classes", 1960, 5870))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.4, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

ck = pd.read_csv(f"{R}/part2_per_check.csv")
ck = ck[ck.curriculum == "original"]
assert ck.seed.nunique() == 20
g = ck.groupby("step").agg(d_mean=("deaths", "mean"), d_sd=("deaths", lambda x: x.std(ddof=0)),
                           a_mean=("alive_before", "mean"), a_sd=("alive_before", lambda x: x.std(ddof=0)))

# ---- verification against the values quoted in the paper ----
EXP_D = {900: 5.10, 1050: 36.85, 2100: 7.20, 2250: 20.60, 2400: 3.15}
EXP_A = {900: 71.1, 1050: 71.0, 2100: 42.0, 2250: 36.8, 2400: 19.6}
for s, v in EXP_D.items():
    assert abs(g.d_mean[s] - v) < 0.005, (s, g.d_mean[s], v)
for s, v in EXP_A.items():
    assert abs(g.a_mean[s] - v) <= 0.051, (s, g.a_mean[s], v)
print("check means verified:", {s: round(g.d_mean[s], 2) for s in EXP_D})
print("alive (before check) verified:", {s: round(g.a_mean[s], 2) for s in EXP_A})


def common(ax, top_labels=True):
    for b in BOUNDS:
        ax.axvline(b + 0.5, color="k", ls=":", lw=0.8, zorder=1)
    tr = blend(ax.transData, ax.transAxes)
    ax.vlines(g.index, 0, 0.025, transform=tr, color=GRAY, lw=0.4, alpha=0.6, zorder=1)  # check ticks
    if top_labels:
        for lab, a, b in PHASES:
            ax.text((a + b) / 2, 1.02, lab, transform=tr, ha="center", va="bottom", fontsize=7, linespacing=0.95)
    ax.set_xlim(0, 5900); ax.set_xlabel("Global training step")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print("saved", f"{FIG}/{name}.pdf")


# ---------------- Figure 1: deaths per check ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for i, (a, b) in enumerate(COUNTED):
    ax.axvspan(a, b, color=GRAY, alpha=0.22, lw=0, zorder=0)
ax.fill_between(g.index, g.d_mean - g.d_sd, g.d_mean + g.d_sd, color=BLUE_FILL, alpha=0.75, lw=0, zorder=2)
ax.plot(g.index, g.d_mean, color=BLUE, lw=0.9, marker="o", ms=2.2, zorder=3)
common(ax)
ax.set_ylabel("Neurons removed per check")
ax.set_ylim(0, 75)
ax.annotate("4→6 burst: 36.85\n(step 1050, after\ncounted window)", xy=(1080, 37.5), xytext=(2450, 50),
            fontsize=6.5, ha="left", va="center", arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
ax.annotate("6→10 burst: 20.60\n(step 2250, inside\ncounted window)", xy=(2280, 21.5), xytext=(2950, 30),
            fontsize=6.5, ha="left", va="center", arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
ax.text(3150, 66, "shaded: counted\n(original metric)", fontsize=6.5, ha="left", va="center", color="#555555")
ax.annotate("", xy=(2200, 68), xytext=(3120, 66), arrowprops=dict(arrowstyle="->", lw=0.5, color="#555555"))
ax.annotate("", xy=(905, 71), xytext=(3120, 67), arrowprops=dict(arrowstyle="->", lw=0.5, color="#555555",
                                                                connectionstyle="arc3,rad=0.12"))
save(fig, "deaths_per_check_timeline")

# ---------------- Figure 2: alive neurons ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.fill_between(g.index, g.a_mean - g.a_sd, g.a_mean + g.a_sd, color=BLUE_FILL, alpha=0.75, lw=0, zorder=2)
ax.plot(g.index, g.a_mean, color=BLUE, lw=0.9, marker="o", ms=2.2, zorder=3)
ax.axhline(16, color="k", ls="--", lw=0.7)
ax.text(2500, 13.5, r"$N_{\min}=16$", fontsize=7, ha="left", va="top")
common(ax)
ax.set_ylabel("Alive neurons\n(entering each check)")
ax.set_ylim(0, 140)
save(fig, "alive_vs_step")

# ---------------- Figure 3: per-seed scatter ----------------
ps = ck.groupby("seed").first()[[]]
ep = pd.read_csv(f"{R}/part2_per_epoch.csv"); ep = ep[(ep.curriculum == "original") & (ep.epoch == 0)]
orig_m = ep[ep.phase == "Medium"].set_index("seed").deaths
orig_h = ep[ep.phase == "Hard"].set_index("seed").deaths
win = lambda ph: ck[(ck.phase == ph) & (ck.step_in_phase <= 300)].groupby("seed").deaths.sum()
w_m, w_h = win("Medium"), win("Hard")
seeds = sorted(orig_m.index)
p1 = pd.read_csv(f"{R}/part1_per_seed.csv").set_index("seed")       # cross-check vs Part 1
assert (p1.loc[seeds, "easy_to_med"] == orig_m[seeds]).all() and (p1.loc[seeds, "med_to_hard"] == orig_h[seeds]).all()

fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4))
for ax, (xm, yh, title, lim) in zip(axes, [
        (orig_m, orig_h, "Original metric\n(first epoch)", 50),
        (w_m, w_h, "Equal window\n(first 300 steps)", 70)]):
    above = int((yh[seeds] > xm[seeds]).sum()); below = int((yh[seeds] < xm[seeds]).sum())
    ties = len(seeds) - above - below
    print(f"{title.splitlines()[0]}: above {above}, below {below}, on diagonal {ties}")
    ax.plot([-2, lim], [-2, lim], ls="--", color=GRAY, lw=0.7, zorder=1)
    for batch, mk, fc in ((B1, "o", BLUE), ([s for s in seeds if s not in B1], "s", "white")):
        ax.scatter(xm[batch], yh[batch], marker=mk, s=14, facecolor=fc, edgecolor=BLUE, lw=0.7,
                   alpha=0.9, zorder=3, label="batch 1" if mk == "o" else "batch 2")
    ax.set_xlim(-2, lim); ax.set_ylim(-2, lim); ax.set_aspect("equal")
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xlabel("Deaths after 4→6")
    ax.text(0.45, 0.96, f"above: {above}", transform=ax.transAxes, fontsize=6.5, va="top", ha="right")
    ax.text(0.96, 0.04, f"below: {below}", transform=ax.transAxes, fontsize=6.5, ha="right")
    ax.tick_params(labelsize=6.5)
    globals().setdefault("COUNTS", []).append((above, below))
assert COUNTS == [(20, 0), (2, 18)], COUNTS
axes[0].set_ylabel("Deaths after 6→10")
axes[1].legend(loc="upper right", frameon=False, fontsize=6, handletextpad=0.2, borderaxespad=0.2)
fig.tight_layout(w_pad=1.0)
save(fig, "perseed_scatter")
