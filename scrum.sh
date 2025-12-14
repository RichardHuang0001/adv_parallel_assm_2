#!/bin/bash

# --- Check for correct number of arguments ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <program> <processors> <problem_size>"
    echo "Example: $0 floyd_chk 16 3000"
    exit 1
fi

# --- Assign command-line arguments to variables ---
PROGRAM=$1
PROCESSORS=$2
PROBLEM_SIZE=$3

# --- Validate processors (must be one of the required perfect squares) ---
case "$PROCESSORS" in
  1)
    NUM_NODES=1
    TASKS_PER_NODE=1
    ;;
  4)
    NUM_NODES=2
    TASKS_PER_NODE=2
    ;;
  9)
    NUM_NODES=3
    TASKS_PER_NODE=3
    ;;
  16)
    NUM_NODES=4
    TASKS_PER_NODE=4
    ;;
  25)
    NUM_NODES=4
    TASKS_PER_NODE=7
    ;;
  36)
    NUM_NODES=4
    TASKS_PER_NODE=9
    ;;
  *)
    echo "Error: <processors> must be one of: 1 4 9 16 25 36"
    exit 1
    ;;
esac

# --- Define job-specific variables ---
JOB_NAME="${PROGRAM}_p${PROCESSORS}_n${PROBLEM_SIZE}"
OUTPUT_FILE="${PROGRAM}_p${PROCESSORS}_n${PROBLEM_SIZE}_%j.out"

# --- Output directory (must exist) ---
OUTDIR="/uac/msc/whuang25/cmsc5702"
mkdir -p "${OUTDIR}"

# --- Create the Slurm job script using a heredoc ---
cat <<EOF > "${PROGRAM}.job"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTDIR}/${OUTPUT_FILE}
#SBATCH --error=${OUTDIR}/%x_%j.err
#SBATCH --mail-user=whuang25@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --chdir=/uac/msc/whuang25/adv_parallel_assm_2

#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks=${PROCESSORS}
#SBATCH --ntasks-per-node=${TASKS_PER_NODE}
#SBATCH --cpus-per-task=1

set -euo pipefail

# Create a hostfile based on SLURM allocated nodes (unique per job)
HOST_FILE="hosts_\${SLURM_JOB_ID}.txt"
scontrol show hostnames "\${SLURM_NODELIST}" | awk '{print \$0" slots=${TASKS_PER_NODE}"}' > "\${HOST_FILE}"

# Run the MPI program
mpiexec.openmpi --hostfile "${HOST_FILE}" -n ${PROCESSORS} ./"${PROGRAM}" /uac/msc/whuang25/adv_parallel_assm_2/data/data${PROBLEM_SIZE}

# Clean up the hostfile
rm -f "\${HOST_FILE}"
EOF

# --- Submit the generated job script ---
sbatch -p cmsc5702_hpc -q cmsc5702 "${PROGRAM}.job"