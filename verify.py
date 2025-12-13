#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple, Optional

INT_RE = re.compile(r"-?\d+")

def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def extract_last_matrix(text: str, n: int) -> List[List[int]]:
    """
    Extract the *last* n-by-n matrix printed in `text`.
    Strategy:
      - For each line, parse all integers.
      - Keep lines whose integer count is exactly n (these are matrix rows).
      - Collect all such rows; take the last n rows as the final matrix.
    This is robust to different spacing/format and ignores timing lines.
    """
    rows: List[List[int]] = []
    for line in text.splitlines():
        nums = [int(x) for x in INT_RE.findall(line)]
        if len(nums) == n:
            rows.append(nums)

    if len(rows) < n:
        raise ValueError(f"Not enough matrix rows found: found {len(rows)} rows of length {n}, need at least {n}.")

    return rows[-n:]

def read_n_from_binary_matrix_file(path: str) -> int:
    # file format: first two ints (m,n) in binary, little-endian typically on x86_64
    # We avoid numpy to keep it simple.
    import struct
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"File too short to contain (m,n): {path}")
        m, n = struct.unpack("ii", header)
    if m != n:
        raise ValueError(f"Matrix must be square but got m={m}, n={n} in {path}")
    return n

def compare_matrices(A: List[List[int]], B: List[List[int]]) -> Optional[Tuple[int,int,int,int]]:
    """
    Return first mismatch (i,j,Aij,Bij) or None if equal.
    """
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != B[i][j]:
                return (i, j, A[i][j], B[i][j])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Directory containing test matrices (default: data)")
    ap.add_argument("--seq", default="./floyd_seq_db", help="Sequential debug executable (default: ./floyd_seq_db)")
    ap.add_argument("--mpi", default="./floyd_chk_db", help="MPI debug executable (default: ./floyd_chk_db)")
    ap.add_argument("--mpirun", default="/usr/local/openmpi/bin/mpirun", help="mpirun path")
    ap.add_argument("--np", type=int, default=4, help="Number of MPI processes (must be perfect square), default=4")
    ap.add_argument("--pattern", default="data*", help="Glob-like prefix filter in data_dir (default: data*)")
    args = ap.parse_args()

    # List files
    data_files = []
    for name in sorted(os.listdir(args.data_dir)):
        if args.pattern.endswith("*"):
            if not name.startswith(args.pattern[:-1]):
                continue
        else:
            if name != args.pattern:
                continue
        full = os.path.join(args.data_dir, name)
        if os.path.isfile(full):
            data_files.append(full)

    if not data_files:
        print(f"No data files found in {args.data_dir} with pattern {args.pattern}", file=sys.stderr)
        sys.exit(2)

    ok = 0
    bad = 0

    for fpath in data_files:
        try:
            n = read_n_from_binary_matrix_file(fpath)
        except Exception as e:
            print(f"[SKIP] {fpath}: cannot read n ({e})")
            continue

        # Run sequential
        rc1, out1, err1 = run_cmd([args.seq, fpath])
        if rc1 != 0:
            print(f"[FAIL] {os.path.basename(fpath)}: sequential rc={rc1}\n{err1}")
            bad += 1
            continue

        # Run MPI
        rc2, out2, err2 = run_cmd([args.mpirun, "-np", str(args.np), args.mpi, fpath])
        if rc2 != 0:
            print(f"[FAIL] {os.path.basename(fpath)}: mpi rc={rc2}\n{err2}")
            bad += 1
            continue

        # Extract final matrices
        try:
            A = extract_last_matrix(out1, n)
            B = extract_last_matrix(out2, n)
        except Exception as e:
            print(f"[FAIL] {os.path.basename(fpath)}: parse error: {e}")
            bad += 1
            continue

        mm = compare_matrices(A, B)
        if mm is None:
            print(f"[PASS] {os.path.basename(fpath)} (n={n}, np={args.np})")
            ok += 1
        else:
            i, j, aij, bij = mm
            print(f"[MISMATCH] {os.path.basename(fpath)} (n={n}, np={args.np}) first diff at (i={i}, j={j}): seq={aij}, mpi={bij}")
            bad += 1

    print(f"\nSummary: PASS={ok}, FAIL={bad}")
    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    main()