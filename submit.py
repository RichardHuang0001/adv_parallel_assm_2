#!/usr/bin/env python3
"""One-stop Slurm runner + result collector for CMSC5702 A2.

Features:
- Submits all required (prog, p, n) combinations (optionally multiple repetitions).
- Enforces max inflight = 2 (cluster policy).
- Monitors jobs via squeue/sacct.
- Parses the printed "<num> seconds" timing from .out.
- Writes an *append-only* long-format CSV record immediately after each job finishes,
  so an interruption won't lose completed results.
- Can resume from an existing CSV (skips already-collected successful runs).
- At the end, writes per-program wide tables (n rows, p columns) using the chosen statistic.

NOTE: This script assumes your scrum.sh creates output files named like:
  {prog}_p{p}_n{n}_{jobid}.out and .err in outdir
and that the program prints a line containing "<num> seconds".
"""

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

OUTDIR_DEFAULT = "/uac/msc/whuang25/cmsc5702"

P_LIST_DEFAULT = [1, 4, 9, 16, 25, 36]
N_LIST_DEFAULT = [60, 480, 1500, 3000, 4020, 6000]
PROGS_DEFAULT = ["floyd_row", "floyd_chk"]

JOBID_RE = re.compile(r"Submitted batch job (\d+)")
TIME_RE = re.compile(r"(?P<sec>[0-9]*\.?[0-9]+)\s*seconds", re.IGNORECASE)

# Some Slurm sites return state values like "COMPLETED", "COMPLETED+",
# or even multiple steps (jobid.batch, jobid.0) with separate lines.
# We'll treat any state that begins with COMPLETED as success.
SUCCESS_STATE_PREFIXES = ("COMPLETED",)


@dataclass(frozen=True)
class Task:
    prog: str
    p: int
    n: int
    rep: int  # repetition index: 1..reps


def run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def submit(task: Task, scrum_path: str) -> int:
    # scrum.sh is expected to accept: prog p n
    rc, out, err = run([scrum_path, task.prog, str(task.p), str(task.n)])
    if rc != 0:
        raise RuntimeError(f"Submit failed for {task}: rc={rc}, err={err}")
    m = JOBID_RE.search(out)
    if not m:
        raise RuntimeError(f"Cannot parse job id for {task}. Output: {out}")
    return int(m.group(1))


def in_queue(jobid: int) -> bool:
    rc, out, _ = run(["squeue", "-h", "-j", str(jobid), "-o", "%i"])
    return rc == 0 and out.strip() == str(jobid)


def sacct_state(jobid: int) -> Optional[str]:
    # Ask for State only; -P makes it parseable. We pick the first non-empty state line.
    rc, out, _ = run(["sacct", "-j", str(jobid), "--format=State", "-n", "-P"])
    if rc != 0 or not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        state = line.split("|")[0].strip()
        if state:
            return state
    return None


def find_out_err(outdir: str, task: Task, jobid: int) -> Tuple[Optional[str], Optional[str]]:
    out_path = os.path.join(outdir, f"{task.prog}_p{task.p}_n{task.n}_{jobid}.out")
    err_path = os.path.join(outdir, f"{task.prog}_p{task.p}_n{task.n}_{jobid}.err")
    return (
        out_path if os.path.exists(out_path) else None,
        err_path if os.path.exists(err_path) else None,
    )


def parse_seconds_from_out(out_path: str) -> Optional[float]:
    try:
        with open(out_path, "r", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return None
    m = TIME_RE.search(txt)
    if not m:
        return None
    return float(m.group("sec"))


def check_result(outdir: str, task: Task, jobid: int, wait_sec: int = 60) -> Tuple[bool, str, Optional[float]]:
    """Validate job output and extract timing.

    Returns: (ok, message, seconds)
    """
    # Slurm may take time to flush output files after the job leaves squeue.
    deadline = time.time() + wait_sec
    out_path, err_path = find_out_err(outdir, task, jobid)
    while not out_path and time.time() < deadline:
        time.sleep(2)
        out_path, err_path = find_out_err(outdir, task, jobid)

    if not out_path:
        return False, "missing .out", None

    sec = parse_seconds_from_out(out_path)

    err_note = ""
    if err_path and os.path.exists(err_path):
        try:
            sz = os.path.getsize(err_path)
        except OSError:
            sz = 0
        if sz != 0:
            err_note = f"non-empty .err ({sz} bytes)"

    if sec is None:
        msg = "no parsable '<num> seconds' line in .out"
        if err_note:
            msg += f"; {err_note}"
        return False, msg, None

    msg = "OK"
    if err_note:
        msg += f" (warning: {err_note})"

    return True, msg, sec


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    # Append-only, flush immediately for crash-safety.
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_completed_ok(csv_path: str) -> Set[Tuple[str, int, int, int]]:
    """Return set of (prog, p, n, rep) that are already successfully collected."""
    done: Set[Tuple[str, int, int, int]] = set()
    if not csv_path or not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ok = str(row.get("ok", "")).strip().lower() in ("true", "1", "yes")
                    if not ok:
                        continue
                    prog = str(row["prog"])
                    p = int(row["p"])
                    n = int(row["n"])
                    rep = int(row.get("rep", "1"))
                    done.add((prog, p, n, rep))
                except Exception:
                    # Ignore malformed lines
                    continue
    except OSError:
        return done
    return done


def stat_aggregate(values: List[float], stat: str) -> Optional[float]:
    if not values:
        return None
    if stat == "mean":
        return sum(values) / len(values)
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    if stat == "median":
        vs = sorted(values)
        mid = len(vs) // 2
        return vs[mid] if len(vs) % 2 == 1 else (vs[mid - 1] + vs[mid]) / 2
    raise ValueError(f"Unknown stat: {stat}")


def write_wide_tables(long_csv: str, out_prefix: str, progs: List[str], p_list: List[int], n_list: List[int], stat: str) -> None:
    """Generate one wide table CSV per program: rows=n, cols=p, value=aggregated seconds."""
    # Load all successful timings from long CSV
    timings: Dict[Tuple[str, int, int], List[float]] = {}
    if not os.path.exists(long_csv):
        return
    with open(long_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ok = str(row.get("ok", "")).strip().lower() in ("true", "1", "yes")
                if not ok:
                    continue
                prog = str(row["prog"])
                p = int(row["p"])
                n = int(row["n"])
                sec = float(row["seconds"])
                timings.setdefault((prog, p, n), []).append(sec)
            except Exception:
                continue

    for prog in progs:
        wide_path = f"{out_prefix}_{prog}_{stat}.csv"
        header = ["n"] + [str(p) for p in p_list]
        with open(wide_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Execution time (seconds)"])
            w.writerow(header)
            for n in n_list:
                row = [n]
                for p in p_list:
                    vals = timings.get((prog, p, n), [])
                    agg = stat_aggregate(vals, stat)
                    row.append("" if agg is None else f"{agg:.6f}")
                w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--scrum", default="./scrum.sh", help="Path to scrum.sh (default: ./scrum.sh)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT, help=f"Output directory (default: {OUTDIR_DEFAULT})")

    ap.add_argument("--progs", nargs="*", default=PROGS_DEFAULT, help="Programs to run")
    ap.add_argument("--p", nargs="*", type=int, default=P_LIST_DEFAULT, help="Process counts")
    ap.add_argument("--n", nargs="*", type=int, default=N_LIST_DEFAULT, help="Problem sizes")

    # Cluster policy: only 2 inflight jobs allowed.
    ap.add_argument("--max_inflight", type=int, default=2, help="Max running/pending jobs at once (MUST be 2)")

    ap.add_argument("--poll_sec", type=int, default=10, help="Polling interval seconds")

    # Repetitions and aggregation statistic
    ap.add_argument("--reps", type=int, default=1, help="Repetitions per (prog,p,n). Use 3+ for averaging.")
    ap.add_argument("--stat", choices=["mean", "min", "max", "median"], default="mean",
                    help="Statistic used for final wide tables")

    # Long-format CSV (append-only). This is the crash-safe log.
    ap.add_argument("--csv", default="timings_long.csv", help="Append-only long CSV path")
    ap.add_argument("--resume", action="store_true", help="Skip tasks already recorded OK in --csv")

    # Output prefix for the final wide tables
    ap.add_argument("--wide_prefix", default="timings_wide", help="Prefix for per-prog wide CSV tables")

    # Retries for failed tasks
    ap.add_argument("--retries", type=int, default=1, help="Retries for failed tasks (default: 1)")

    args = ap.parse_args()

    if args.max_inflight != 2:
        print("[WARN] Cluster policy: max_inflight must be 2. Forcing to 2.")
        args.max_inflight = 2

    if not os.path.exists(args.scrum):
        raise SystemExit(f"Cannot find {args.scrum}. Run from project dir or pass --scrum.")

    os.makedirs(args.outdir, exist_ok=True)

    fieldnames = ["timestamp", "jobid", "state", "prog", "p", "n", "rep", "seconds", "ok", "note"]
    ensure_csv_header(args.csv, fieldnames)

    completed_ok: Set[Tuple[str, int, int, int]] = load_completed_ok(args.csv) if args.resume else set()
    if args.resume and completed_ok:
        print(f"[RESUME] Found {len(completed_ok)} successful runs in {args.csv}; will skip them.")

    # Build task list in a stable order: prog -> n -> p -> rep
    tasks: List[Task] = []
    for prog in args.progs:
        for n in args.n:
            for p in args.p:
                for rep in range(1, args.reps + 1):
                    t = Task(prog, p, n, rep)
                    if args.resume and (prog, p, n, rep) in completed_ok:
                        continue
                    tasks.append(t)

    # Track how many times we've tried each task
    attempts: Dict[Task, int] = {t: 0 for t in tasks}

    pending: List[Task] = tasks[:]
    running: Dict[int, Task] = {}

    finished_ok: List[Tuple[int, Task]] = []
    finished_bad: List[Tuple[int, Task, str]] = []

    print(f"Total tasks to run: {len(tasks)} (after resume filtering)")
    print(f"Inflight limit: {args.max_inflight}")
    print(f"Outdir: {args.outdir}")
    print(f"Long CSV: {args.csv}")
    print(f"Reps: {args.reps}; final stat for wide tables: {args.stat}")

    while pending or running:
        # Submit up to inflight limit
        while pending and len(running) < args.max_inflight:
            t = pending.pop(0)
            attempts[t] += 1
            try:
                jobid = submit(t, args.scrum)
            except Exception as e:
                # Submission failed: retry if allowed
                if attempts[t] <= args.retries + 1:
                    print(f"[SUBMIT FAIL] {t} -> {e}. Will retry.")
                    pending.append(t)
                    time.sleep(2)
                    continue
                else:
                    print(f"[SUBMIT FAIL] {t} -> {e}. Giving up.")
                    finished_bad.append((-1, t, f"submit failed: {e}"))
                    # Log failure immediately
                    append_csv_row(args.csv, fieldnames, {
                        "timestamp": int(time.time()),
                        "jobid": -1,
                        "state": "SUBMIT_FAILED",
                        "prog": t.prog,
                        "p": t.p,
                        "n": t.n,
                        "rep": t.rep,
                        "seconds": "",
                        "ok": False,
                        "note": f"submit failed: {e}",
                    })
                    continue

            running[jobid] = t
            print(f"[SUBMIT] job {jobid}: {t.prog} p={t.p} n={t.n} rep={t.rep} (attempt {attempts[t]})")

        # Poll running jobs
        done_jobids: List[int] = []
        for jobid, t in list(running.items()):
            if in_queue(jobid):
                continue

            state = sacct_state(jobid) or "UNKNOWN"
            ok_out, msg, sec = check_result(args.outdir, t, jobid)

            # Require COMPLETED* state for a run to count as OK
            ok = ok_out and state.startswith(SUCCESS_STATE_PREFIXES)
            if not ok and ok_out and not state.startswith(SUCCESS_STATE_PREFIXES):
                msg = f"state={state}, {msg}"

            # Append record immediately for crash-safety
            append_csv_row(args.csv, fieldnames, {
                "timestamp": int(time.time()),
                "jobid": jobid,
                "state": state,
                "prog": t.prog,
                "p": t.p,
                "n": t.n,
                "rep": t.rep,
                "seconds": "" if sec is None else f"{sec:.6f}",
                "ok": ok,
                "note": msg,
            })

            if ok:
                finished_ok.append((jobid, t))
                print(f"[DONE OK] job {jobid} state={state}: {t.prog} p={t.p} n={t.n} rep={t.rep} time={sec:.6f}s")
            else:
                # Retry if allowed
                if attempts[t] <= args.retries:
                    print(f"[DONE BAD] job {jobid} {t.prog} p={t.p} n={t.n} rep={t.rep} -> {state}, {msg} (will retry)")
                    pending.append(t)
                else:
                    finished_bad.append((jobid, t, f"state={state}, {msg}"))
                    print(f"[DONE BAD] job {jobid} {t.prog} p={t.p} n={t.n} rep={t.rep} -> {state}, {msg} (giving up)")

            done_jobids.append(jobid)

        for jobid in done_jobids:
            running.pop(jobid, None)

        print(f"[PROGRESS] inflight={len(running)} pending={len(pending)} ok={len(finished_ok)} bad={len(finished_bad)}")
        if pending or running:
            time.sleep(args.poll_sec)

    # Generate wide tables at the end for easy copy into the report/Excel.
    write_wide_tables(
        long_csv=args.csv,
        out_prefix=args.wide_prefix,
        progs=args.progs,
        p_list=args.p,
        n_list=args.n,
        stat=args.stat,
    )
    print(f"\n[WIDE] Wrote per-program wide tables: {args.wide_prefix}_<prog>_{args.stat}.csv")

    print("\n=== Summary ===")
    print(f"OK:  {len(finished_ok)}")
    print(f"BAD: {len(finished_bad)}")
    if finished_bad:
        print("\nFailures:")
        for jobid, t, reason in finished_bad:
            print(f"  job {jobid}: {t.prog} p={t.p} n={t.n} rep={t.rep} -> {reason}")


if __name__ == "__main__":
    main()
