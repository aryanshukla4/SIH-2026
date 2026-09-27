#!/usr/bin/env python3
"""Figures and a README table from a `bench_suite.py` CSV.

    python scripts/plot_bench.py results/netlib.csv [more.csv ...] [--out DIR]

Writes PNGs sized for a 16:9 slide (and legible on GitHub) plus
`summary.md`, a markdown table ready to paste into the README.

SCORING. Speed only counts for a CORRECT answer:
  * feasible set  -- status Optimal AND objective within REL_TOL of the
    reference (HiGHS if it solved, else SoPlex, else Clp);
  * infeasible set -- status Infeasible (InfeasibleOrUnbounded is not a
    certificate of infeasibility and is counted as no verdict).
A wrong or missing answer is charged the time limit, as in every published
solver comparison (Mittelmann; the cuPDLPx paper's SGM10).

Needs pandas and matplotlib.
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REL_TOL = 1e-6
REFERENCE_ORDER = ["highs", "soplex", "clp"]

# The deck's own colours (sovsolve-SIH2026.pptx), so figures drop straight in.
INK, MUTED, GRID, SURFACE = "#232B36", "#555F6D", "#D9DEE5", "#FFFFFF"
OURS = "#0F6CB2"
COLOR = {
    "sovsolve": OURS, "highs": "#C95D0A", "highs-pdlp": "#E8A33D",
    "highs-ipm": "#8C3B00", "clp": "#1B7A3D", "soplex": "#6B4FBB",
    "scip": "#808080",
    "sovsolve-dual": "#5B8DB8", "sovsolve-primal": "#4F9BD9",
    "sovsolve-pdlpx": "#1F3864", "sovsolve-hsd": "#8DB9E2",
}
LABEL = {
    "sovsolve": "sovsolve (concurrent)", "highs": "HiGHS", "highs-pdlp": "HiGHS PDLP",
    "highs-ipm": "HiGHS IPM", "clp": "Clp (COIN-OR)", "soplex": "SoPlex",
    "scip": "SCIP", "sovsolve-dual": "dual simplex", "sovsolve-primal": "primal simplex",
    "sovsolve-pdlpx": "pdlpx", "sovsolve-hsd": "HSD",
}
DASH = {"highs-pdlp": (0, (5, 2)), "scip": (0, (2, 2)), "highs-ipm": (0, (6, 2, 1, 2))}
ENGINE_NAME = {"dual-simplex": "sovsolve-dual", "primal-simplex": "sovsolve-primal",
               "pdlpx": "sovsolve-pdlpx", "hsd": "sovsolve-hsd"}

_FONTS = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams.update({
    # Calibri is the deck's face; Carlito is its metric twin on Linux.
    "font.family": [f for f in ("Calibri", "Carlito") if f in _FONTS] + ["DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def label(s):
    return LABEL.get(s, s)


def load(paths):
    df = pd.concat([pd.read_csv(p, dtype={"git_commit": str}) for p in paths], ignore_index=True)
    df = df[df["status"] != "Skipped"].copy()
    df["wall_seconds"] = pd.to_numeric(df["wall_seconds"], errors="coerce")
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    limit = float(pd.to_numeric(df["time_limit"], errors="coerce").max())

    ref = {}
    for (set_name, inst), g in df.groupby(["set", "instance"]):
        for solver in REFERENCE_ORDER:
            hit = g[(g["solver"] == solver) & (g["status"] == "Optimal")]
            if len(hit):
                ref[(set_name, inst)] = hit["objective"].iloc[0]
                break

    def correct(row):
        expected = row.get("expected")
        if expected == "Infeasible":
            return row["status"] == "Infeasible"
        if row["status"] != "Optimal":
            return False
        r = ref.get((row["set"], row["instance"]))
        if r is None or pd.isna(row["objective"]):
            return expected != "Optimal"  # no reference: trust the verdict
        return abs(row["objective"] - r) <= REL_TOL * max(1.0, abs(r))

    df["correct"] = df.apply(correct, axis=1)
    df["charged"] = df["wall_seconds"].where(df["correct"], limit).fillna(limit)
    return df, limit


def sgm10(times):
    return math.exp(sum(math.log(t + 10.0) for t in times) / len(times)) - 10.0


def main_lineup(df):
    order = ["sovsolve", "highs", "highs-pdlp", "highs-ipm", "clp", "soplex", "scip"]
    present = set(df["solver"])
    return [s for s in order if s in present]


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("  " + path)


def footer(fig, df):
    r = df.iloc[0]
    # Anchored just BELOW the canvas, so it never collides with the x label.
    fig.text(0.0, 0.0, "%s · %s cores · %s · CPU only · time limit %ss · wall clock incl. file read"
             % (str(r.get("machine_label") or r.get("cpu") or "")[:60], r.get("cores", "?"),
                r.get("os", ""), int(float(r.get("time_limit", 0)))),
             fontsize=10, color=MUTED, ha="left", va="top",
             transform=matplotlib.transforms.offset_copy(
                 fig.transFigure, fig=fig, y=-6, units="points"))


def perf_profile(df, solvers, out, set_name):
    """Dolan & More (2002): for each solver, the fraction of models it solves
    within a factor tau of the fastest solver on that model."""
    d = df[df["set"] == set_name]
    if d.empty:
        return
    t = d.pivot_table(index="instance", columns="solver", values="charged", aggfunc="min")
    ok = d.pivot_table(index="instance", columns="solver", values="correct", aggfunc="max")
    t, ok = t[[s for s in solvers if s in t]], ok[[s for s in solvers if s in ok]]
    best = t.where(ok.astype(bool)).min(axis=1)
    keep = best.notna()
    ratios = t[keep].div(best[keep], axis=0).where(ok[keep].astype(bool))
    tau_max = max(2.0, float(ratios.max().max() or 2)) * 1.5
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in ratios.columns:
        r = ratios[s].dropna().sort_values().clip(lower=1.0).tolist()
        n = len(ratios)
        xs, ys = [1.0], [0.0]
        for i, v in enumerate(r):
            xs += [v, v]
            ys += [ys[-1], (i + 1) / n]
        xs.append(tau_max)
        ys.append(ys[-1])
        ax.plot(xs, ys, color=COLOR.get(s), linewidth=3.2 if s == "sovsolve" else 2,
                linestyle=DASH.get(s, "-"), zorder=3 if s == "sovsolve" else 2,
                # Final share in the legend, not at the line ends: ties at
                # 100% would stack the labels on top of each other.
                label="%s — %d%%" % (label(s), round(100 * ys[-1])))
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, tau_max)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("within this factor of the fastest solver (log scale)")
    ax.set_ylabel("models solved correctly")
    ax.set_title("Performance profile — Netlib %s (%d models)" % (set_name, len(ratios)))
    ax.legend(loc="lower right")
    footer(fig, d)
    save(fig, out, "perf_profile_%s.png" % set_name)


def sgm_bars(df, solvers, out, set_name, limit):
    d = df[(df["set"] == set_name) & df["solver"].isin(solvers)]
    if d.empty:
        return []
    rows = []
    for s, g in d.groupby("solver"):
        best = g.groupby("instance")["charged"].min()
        rows.append((s, sgm10(best.tolist()), int(g.groupby("instance")["correct"].max().sum()),
                     g["instance"].nunique()))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(rows) + 1.5))
    names = [label(r[0]) for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(names, vals, color=[COLOR.get(r[0]) for r in rows], height=0.6)
    for i, (s, v, solved, n) in enumerate(rows):
        ax.text(v, i, "  %.3f s   ·   %d/%d correct" % (v, solved, n), va="center",
                color=INK, fontsize=12, fontweight="bold" if s == "sovsolve" else "normal")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.6)
    ax.set_xlabel("shifted geometric mean of wall time, shift 10 s (lower is better)")
    ax.set_title("SGM10 solve time — Netlib %s" % set_name.replace(
        "-engines", ": the race vs each engine alone"))
    footer(fig, d)
    save(fig, out, "sgm10_%s.png" % set_name)
    return rows


def solved_over_time(df, solvers, out, set_name, limit):
    d = df[(df["set"] == set_name) & df["solver"].isin(solvers)]
    if d.empty:
        return
    n = d["instance"].nunique()
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in solvers:
        g = d[(d["solver"] == s) & d["correct"]].groupby("instance")["wall_seconds"].min()
        ts = sorted(max(t, 1e-3) for t in g)
        if not ts:
            continue
        xs = [1e-3] + [x for t in ts for x in (t, t)] + [limit]
        ys = [0] + [y for i in range(len(ts)) for y in (i, i + 1)] + [len(ts)]
        ax.plot(xs, ys, color=COLOR.get(s), linestyle=DASH.get(s, "-"),
                linewidth=3.2 if s == "sovsolve" else 2, label="%s (%d)" % (label(s), len(ts)))
    ax.axhline(n, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-3 * 5, limit)
    ax.set_ylim(0, n * 1.05)
    ax.set_xlabel("wall time, seconds (log scale)")
    ax.set_ylabel("models solved correctly")
    ax.set_title("Models solved vs time — Netlib %s (%d models)" % (set_name, n))
    ax.legend(loc="upper left")
    footer(fig, d)
    save(fig, out, "solved_vs_time_%s.png" % set_name)


def infeasible_bars(df, solvers, out):
    d = df[(df["set"] == "infeasible") & df["solver"].isin(solvers)]
    if d.empty:
        return []
    rows = []
    for s in solvers:
        g = d[d["solver"] == s].groupby("instance").agg(
            correct=("correct", "max"), status=("status", "first"))
        if g.empty:
            continue
        right = int(g["correct"].sum())
        wrong = int(g["status"].isin(["Optimal", "Unbounded"]).sum())
        rows.append((s, right, wrong, len(g) - right - wrong, len(g)))
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(rows) + 1.5))
    for i, (s, right, wrong, none, n) in enumerate(rows):
        ax.barh(i, right, color=COLOR.get(s), height=0.6)
        ax.barh(i, wrong, left=right, color="#C0392B", height=0.6, hatch="//", edgecolor=SURFACE)
        ax.barh(i, none, left=right + wrong, color=GRID, height=0.6)
        ax.text(n, i, "  %d/%d proved infeasible%s" % (
            right, n, ("  ·  %d wrong verdict" % wrong) if wrong else ""),
            va="center", fontsize=12, fontweight="bold" if s == "sovsolve" else "normal")
    ax.set_yticks(range(len(rows)), [label(r[0]) for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, rows[0][4] * 1.7)
    ax.set_xlabel("models   (colour = proved infeasible · red hatch = wrong verdict · grey = no verdict)")
    ax.set_title("Infeasibility detection — Netlib infeasible set")
    footer(fig, d)
    save(fig, out, "infeasible_detection.png")
    return rows


def head_to_head(df, a, b, out, name):
    d = df[df["solver"].isin([a, b])]
    t = d.pivot_table(index=["set", "instance"], columns="solver", values="charged", aggfunc="min")
    if a not in t or b not in t:
        return
    t = t.dropna()
    fig, ax = plt.subplots(figsize=(7.5, 7))
    lo, hi = max(1e-3, t.min().min() / 1.5), t.max().max() * 1.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    faster = (t[a] < t[b]).sum()
    ax.scatter(t[b], t[a], s=46, color=COLOR.get(a), edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("%s wall time, s" % label(b))
    ax.set_ylabel("%s wall time, s" % label(a))
    ax.text(0.03, 0.97, "below the line: %s faster\n%d of %d models" % (label(a), faster, len(t)),
            transform=ax.transAxes, va="top", fontsize=12, color=INK)
    ax.set_title("%s vs %s" % (label(a), label(b)))
    footer(fig, d)
    save(fig, out, name)


def winners(df, out):
    d = df[(df["solver"] == "sovsolve") & df["correct"]]
    w = d["concurrent_winner"].dropna()
    w = w[w != ""]
    if w.empty:
        return None
    counts = w.map(lambda e: label(ENGINE_NAME.get(e, e))).value_counts()
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(counts) + 1.4))
    ax.barh(counts.index, counts.values, color=OURS, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v, i, "  %d" % v, va="center", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.2)
    ax.set_xlabel("models on which this engine finished first")
    ax.set_title("Which engine wins the race — no single engine wins everywhere")
    footer(fig, d)
    save(fig, out, "concurrent_winners.png")
    return counts


def summary_md(df, out, limit, bars, infeas, counts):
    r = df.iloc[0]
    lines = ["# Benchmark summary", "",
             "Machine: %s — %s cores — %s. CPU only. Time limit %d s. "
             "Commit `%s`. Wall clock, including reading the file." % (
                 r.get("machine_label") or r.get("cpu"), r.get("cores"), r.get("os"),
                 int(limit), r.get("git_commit")), ""]
    for set_name, rows in bars.items():
        if not rows:
            continue
        lines += ["## Netlib %s" % set_name, "",
                  "| Solver | Correct | SGM10 (s) | Relative |", "|---|---:|---:|---:|"]
        base = rows[0][1]
        for s, v, solved, n in rows:
            name = "**%s**" % label(s) if s == "sovsolve" else label(s)
            lines.append("| %s | %d/%d | %.3f | %.2f× |" % (name, solved, n, v, v / base))
        lines.append("")
    if infeas:
        lines += ["## Infeasibility detection", "",
                  "| Solver | Proved infeasible | Wrong verdict | No verdict |",
                  "|---|---:|---:|---:|"]
        for s, right, wrong, none, n in infeas:
            lines.append("| %s | %d/%d | %d | %d |" % (label(s), right, n, wrong, none))
        lines.append("")
    if counts is not None:
        lines += ["## Concurrent race winners", ""]
        lines += ["- %s: %d" % (k, v) for k, v in counts.items()]
        lines.append("")
    path = os.path.join(out, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("  " + path)


def deck_figures(df, out, limit):
    """Charts drawn AT the size of the deck's chart slots (3.58 x 1.72 in),
    so text stays legible instead of being shrunk from a full-page figure.
    The slide carries each chart's title; the figure carries none."""
    deck = os.path.join(out, "deck")
    os.makedirs(deck, exist_ok=True)
    size = (3.58, 1.72)
    small = {"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
             "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titlesize": 7,
             "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
             "lines.linewidth": 1.4}
    solvers = main_lineup(df)
    with plt.rc_context(small):
        # 1. The race: which engine finished first, feasible + infeasible.
        # Every model the race DECIDED counts -- who finished first does not
        # depend on the reference check. Models presolve settles never race.
        d = df[df["solver"] == "sovsolve"]
        w = d["concurrent_winner"].dropna()
        w = w[w != ""]
        unraced = len(d) - len(w)
        w = w.map(lambda e: label(ENGINE_NAME.get(e, e))).value_counts()
        if not w.empty:
            fig, ax = plt.subplots(figsize=size)
            ax.barh(w.index, w.values, color=OURS, height=0.62)
            for i, v in enumerate(w.values):
                ax.text(v, i, " %d" % v, va="center", fontsize=7, color=INK)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, w.max() * 1.18)
            ax.set_xlabel("models won, of %d%s" % (
                len(d), ("; presolve settled %d" % unraced) if unraced else ""))
            save(fig, deck, "1_race_winners.png")

        # 2. Models solved correctly vs time, feasible set.
        f = df[(df["set"] == "feasible") & df["solver"].isin(solvers)]
        if not f.empty:
            n = f["instance"].nunique()
            fig, ax = plt.subplots(figsize=size)
            for s in solvers:
                g = f[(f["solver"] == s) & f["correct"]].groupby("instance")["wall_seconds"].min()
                ts = sorted(max(t, 1e-3) for t in g)
                if not ts:
                    continue
                xs = [1e-3] + [x for t in ts for x in (t, t)] + [limit]
                ys = [0] + [y for i in range(len(ts)) for y in (i, i + 1)] + [len(ts)]
                ax.plot(xs, ys, color=COLOR.get(s), linestyle=DASH.get(s, "-"),
                        linewidth=2.2 if s == "sovsolve" else 1.2,
                        label=label(s).replace(" (concurrent)", ""),
                        zorder=3 if s == "sovsolve" else 2)
            ax.set_xscale("log")
            ax.set_xlim(3e-3, max(20.0, f["wall_seconds"].max() * 1.5))
            ax.set_ylim(0, n * 1.04)
            ax.set_xlabel("wall time, s (log)")
            ax.set_ylabel("solved of %d" % n)
            ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.2)
            save(fig, deck, "2_solved_vs_time.png")

        # 3. Infeasible set: how many each solver PROVED infeasible, and its
        # median time on those. Both matter -- a fast wrong verdict is not fast.
        inf = df[(df["set"] == "infeasible") & df["solver"].isin(solvers)]
        rows = []
        for s in solvers:
            g = inf[inf["solver"] == s]
            if g.empty:
                continue
            ok = g[g["correct"]]
            rows.append((s, len(ok), g["instance"].nunique(),
                         1000 * ok["wall_seconds"].median() if len(ok) else float("nan")))
        if rows:
            rows.sort(key=lambda r: (-r[1], r[3]))
            fig, ax = plt.subplots(figsize=size)
            names = [label(s).replace(" (concurrent)", "") for s, *_ in rows]
            ax.barh(names, [r[3] for r in rows], color=[COLOR.get(r[0]) for r in rows],
                    height=0.62)
            for i, (s, proved, n, ms) in enumerate(rows):
                ax.text(ms, i, "  %d/%d proved · %.1f ms" % (proved, n, ms), va="center",
                        fontsize=7, color=INK, fontweight="bold" if s == "sovsolve" else "normal")
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, max(r[3] for r in rows) * 2.1)
            ax.set_xlabel("median time to prove infeasible, ms")
            save(fig, deck, "3_infeasible.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", help="output directory (default: next to the first CSV)")
    args = ap.parse_args()
    df, limit = load(args.csv)
    out = args.out or os.path.splitext(args.csv[0])[0] + "-figs"
    os.makedirs(out, exist_ok=True)
    solvers = main_lineup(df)
    print("writing figures to %s" % out)

    bars = {}
    for set_name in sorted(set(df["set"])):
        if set_name != "infeasible":
            perf_profile(df, solvers, out, set_name)
            solved_over_time(df, solvers, out, set_name, limit)
        bars[set_name] = sgm_bars(df, solvers, out, set_name, limit)
    infeas = infeasible_bars(df, solvers, out)
    head_to_head(df, "sovsolve", "highs", out, "sovsolve_vs_highs.png")
    head_to_head(df, "sovsolve-pdlpx", "highs-pdlp", out, "pdlpx_vs_highs_pdlp.png")
    singles = [s for s in ENGINE_NAME.values() if s in set(df["solver"])]
    if singles:
        for set_name in sorted(set(df["set"]) - {"infeasible"}):
            bars["%s (our engines)" % set_name] = sgm_bars(
                df.assign(set=df["set"].where(df["set"] != set_name,
                                              "%s-engines" % set_name)),
                ["sovsolve"] + singles, out, "%s-engines" % set_name, limit)
    counts = winners(df, out)
    summary_md(df, out, limit, bars, infeas, counts)
    deck_figures(df, out, limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
