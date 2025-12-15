#!/usr/bin/env python3
import argparse, csv, os
from collections import defaultdict
import matplotlib.pyplot as plt

P_EXPECT = [1, 4, 9, 16, 25, 36]
PROG_TO_TITLE = {"floyd_row": "Striped (Row-wise) MPI", "floyd_chk": "Checkerboard MPI"}

def read_data(path):
    data = defaultdict(lambda: defaultdict(list))  # data[(prog,metric)][n] = [(p,val),...]
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            prog = row["program"].strip()
            n = int(row["n"]); p = int(row["p"])
            sp = float(row["speedup"]); ef = float(row["efficiency"])
            data[(prog,"speedup")][n].append((p, sp))
            data[(prog,"efficiency")][n].append((p, ef))
    for k in data:
        for n in data[k]:
            data[k][n].sort(key=lambda x: x[0])
    return data

def plot_one(data, prog, metric, outdir):
    title_prog = PROG_TO_TITLE.get(prog, prog)
    metric_title = "Relative Speedup" if metric == "speedup" else "Efficiency"
    ylab = "Speedup (T1/Tp)" if metric == "speedup" else "Efficiency (Speedup/p)"
    plt.figure()
    plt.title(f"{title_prog} - {metric_title}")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel(ylab)

    curves = data.get((prog, metric), {})
    for n in sorted(curves.keys()):
        xs = [p for p,_ in curves[n]]
        ys = [v for _,v in curves[n]]
        plt.plot(xs, ys, marker="o", label=f"n={n}")

    plt.xticks(P_EXPECT)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(title="Problem size", fontsize="small")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{prog}_{metric}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="speedup_efficiency.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Cannot find {args.csv} (run from the dir containing it, or pass --csv).")

    data = read_data(args.csv)
    outs = [
        plot_one(data, "floyd_row", "speedup", args.outdir),
        plot_one(data, "floyd_row", "efficiency", args.outdir),
        plot_one(data, "floyd_chk", "speedup", args.outdir),
        plot_one(data, "floyd_chk", "efficiency", args.outdir),
    ]
    print("Generated:")
    for o in outs:
        print("  " + o)

if __name__ == "__main__":
    main()
