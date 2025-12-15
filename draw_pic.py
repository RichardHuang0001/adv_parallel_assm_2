#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

P_LIST = [1, 4, 9, 16, 25, 36]
N_LIST = [60, 480, 1500, 3000, 4020, 6000]

PROG_LABEL = {
    "floyd_row": "Striped (Row-wise)",
    "floyd_chk": "Checkerboard",
}

def read_results_min(path: str):
    """
    Read results_min.csv:
      program,p,n,seconds_min,jobid,file,samples,seconds_all
    Return: t[program][n][p] = seconds_min (float)
    """
    t = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            prog = row["program"].strip()
            p = int(row["p"])
            n = int(row["n"])
            sec = float(row["seconds_min"])
            t[prog][n][p] = sec
    return t

def speedup(t1, tp):
    return t1 / tp

def efficiency(s, p):
    return s / p

def karp_flatt(s, p):
    # e(n,p) = (1/S - 1/p) / (1 - 1/p), p>1
    if p <= 1:
        return float("nan")
    return (1.0/s - 1.0/p) / (1.0 - 1.0/p)

def prep_curves(tprog):
    """
    Given tprog[n][p]=time, compute curves for each n:
    time: list[(p,time)]
    speedup: list[(p,S)]
    eff: list[(p,E)]
    karp: list[(p,e)]
    """
    curves = {}
    for n in N_LIST:
        if n not in tprog:
            continue
        # require T(1)
        if 1 not in tprog[n]:
            continue
        t1 = tprog[n][1]
        time_pts = []
        sp_pts = []
        eff_pts = []
        kf_pts = []
        for p in P_LIST:
            if p not in tprog[n]:
                continue
            tp = tprog[n][p]
            s = speedup(t1, tp)
            e = efficiency(s, p)
            k = karp_flatt(s, p)
            time_pts.append((p, tp))
            sp_pts.append((p, s))
            eff_pts.append((p, e))
            kf_pts.append((p, k))
        curves[n] = {
            "time": time_pts,
            "speedup": sp_pts,
            "eff": eff_pts,
            "karp": kf_pts,
        }
    return curves

def set_x_style(ax):
    ax.set_xlim(0, 40)
    ax.set_xticks(list(range(0, 41, 5)))
    ax.grid(True)

def plot_panel(ax, curves, metric, title, ylabel, eff_percent=False):
    ax.set_title(title)
    ax.set_xlabel("Number of processors")
    ax.set_ylabel(ylabel)
    set_x_style(ax)

    for n in N_LIST:
        if n not in curves:
            continue
        pts = curves[n][metric]
        xs = [p for p, _ in pts]
        ys = []
        for _, v in pts:
            if eff_percent:
                ys.append(v * 100.0)
            else:
                ys.append(v)
        ax.plot(xs, ys, marker="o", label=f"{n}")

    # Legend on the right, like the template
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

def draw_one_program(prog, curves, outdir, time_scale="linear"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(PROG_LABEL.get(prog, prog), fontsize=14)

    # Execution time
    ax = axs[0][0]
    plot_panel(ax, curves, "time", "Execution time", "Execution time (s)")
    if time_scale == "log":
        ax.set_yscale("log")

    # Relative Speedup
    plot_panel(axs[0][1], curves, "speedup", "Relative Speedup", "Speedup")

    # Karp-Flatt Metric
    plot_panel(axs[1][0], curves, "karp", "Karp-Flatt Metric", "e(n, p)")

    # Relative Efficiency (percent)
    ax_eff = axs[1][1]
    plot_panel(ax_eff, curves, "eff", "Relative Efficiency", "Efficiency", eff_percent=True)
    ax_eff.set_ylim(0, 100)
    ax_eff.set_yticks(list(range(0, 101, 10)))
    ax_eff.set_yticklabels([f"{x}%" for x in range(0, 101, 10)])

    # Layout to make room for legends on the right
    plt.tight_layout(rect=[0, 0, 0.86, 0.95])

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{prog}_template_style.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_min.csv", help="Input results_min.csv")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--time_scale", choices=["linear", "log"], default="linear",
                    help="Y scale for execution time plot (default: linear)")
    args = ap.parse_args()

    t = read_results_min(args.results)

    outs = []
    for prog in ["floyd_row", "floyd_chk"]:
        if prog not in t:
            continue
        curves = prep_curves(t[prog])
        outs.append(draw_one_program(prog, curves, args.outdir, time_scale=args.time_scale))

    print("Generated:")
    for o in outs:
        print("  " + o)

if __name__ == "__main__":
    main()
