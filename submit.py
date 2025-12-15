#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

OUTDIR_DEFAULT = "/uac/msc/whuang25/cmsc5702"

P_LIST_DEFAULT = [1, 4, 9, 16, 25, 36]
N_LIST_DEFAULT = [60, 480, 1500, 3000, 4020, 6000]
PROGS_DEFAULT = ["floyd_row", "floyd_chk"]

JOBID_RE = re.compile(r"Submitted batch job (\d+)")

@dataclass(frozen=True)
class Task:
    prog: str
    p: int
    n: int

def run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def submit(task: Task, scrum_path: str) -> int:
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
    rc, out, _ = run(["sacct", "-j", str(jobid), "--format=State", "-n", "-P"])
    if rc != 0 or not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line:
            return line.split("|")[0]
    return None

def find_out_err(outdir: str, task: Task, jobid: int) -> Tuple[Optional[str], Optional[str]]:
    out_path = os.path.join(outdir, f"{task.prog}_p{task.p}_n{task.n}_{jobid}.out")
    err_path = os.path.join(outdir, f"{task.prog}_p{task.p}_n{task.n}_{jobid}.err")
    return (out_path if os.path.exists(out_path) else None,
            err_path if os.path.exists(err_path) else None)

def check_result(outdir: str, task: Task, jobid: int) -> Tuple[bool, str]:
    out_path, err_path = find_out_err(outdir, task, jobid)

    if not out_path:
        return False, "missing .out"

    if err_path and os.path.getsize(err_path) != 0:
        return False, f"non-empty .err ({os.path.getsize(err_path)} bytes)"

    with open(out_path, "r", errors="ignore") as f:
        txt = f.read()
    if "seconds" not in txt:
        return False, "no 'seconds' line in .out"
    return True, "OK"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrum", default="./scrum.sh", help="Path to scrum.sh (default: ./scrum.sh)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT, help=f"Output directory (default: {OUTDIR_DEFAULT})")
    ap.add_argument("--progs", nargs="*", default=PROGS_DEFAULT, help="Programs to run (default: floyd_row floyd_chk)")
    ap.add_argument("--p", nargs="*", type=int, default=P_LIST_DEFAULT, help="Process counts (default: 1 4 9 16 25 36)")
    ap.add_argument("--n", nargs="*", type=int, default=N_LIST_DEFAULT, help="Problem sizes (default: 60 480 1500 3000 4020 6000)")
    ap.add_argument("--max_inflight", type=int, default=8, help="Max running/pending jobs at once (default: 8)")
    ap.add_argument("--poll_sec", type=int, default=10, help="Polling interval seconds (default: 10)")
    args = ap.parse_args()

    if not os.path.exists(args.scrum):
        raise SystemExit(f"Cannot find {args.scrum}. Run from project dir or pass --scrum.")
    os.makedirs(args.outdir, exist_ok=True)

    tasks: List[Task] = [Task(prog, p, n) for n in args.n for p in args.p for prog in args.progs]

    pending: List[Task] = tasks[:]
    running: Dict[int, Task] = {}
    finished_ok: List[Tuple[int, Task]] = []
    finished_bad: List[Tuple[int, Task, str]] = []

    print(f"Total tasks: {len(tasks)}; max inflight={args.max_inflight}")
    print(f"Output dir: {args.outdir}")

    while pending or running:
        while pending and len(running) < args.max_inflight:
            t = pending.pop(0)
            jobid = submit(t, args.scrum)
            running[jobid] = t
            print(f"[SUBMIT] job {jobid}: {t.prog} p={t.p} n={t.n}")

        done_jobids: List[int] = []
        for jobid, t in list(running.items()):
            if in_queue(jobid):
                continue

            state = sacct_state(jobid) or "UNKNOWN"
            ok, msg = check_result(args.outdir, t, jobid)
            if ok:
                finished_ok.append((jobid, t))
                print(f"[DONE OK] job {jobid} state={state}: {t.prog} p={t.p} n={t.n}")
            else:
                finished_bad.append((jobid, t, f"state={state}, {msg}"))
                print(f"[DONE BAD] job {jobid} {t.prog} p={t.p} n={t.n} -> {state}, {msg}")
            done_jobids.append(jobid)

        for jobid in done_jobids:
            running.pop(jobid, None)

        print(f"[PROGRESS] inflight={len(running)} pending={len(pending)} ok={len(finished_ok)} bad={len(finished_bad)}")
        if pending or running:
            time.sleep(args.poll_sec)

    print("\n=== Summary ===")
    print(f"OK:  {len(finished_ok)}")
    print(f"BAD: {len(finished_bad)}")
    if finished_bad:
        print("\nFailures:")
        for jobid, t, reason in finished_bad:
            print(f"  job {jobid}: {t.prog} p={t.p} n={t.n} -> {reason}")

if __name__ == "__main__":
    main()
