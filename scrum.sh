#!/bin/bash

# --- Check for correct number of arguments ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <program> <processors> <problem_size>"
    exit 1
fi

# --- Assign command-line arguments to variables ---
PROGRAM=$1
PROCESSORS=$2
PROBLEM_SIZE=$3

if [ "$PROCESSORS" -eq 1 ]; then
    NUM_NODES=1
    TASKS_PER_NODE=1
elif [ "$PROCESSORS" -eq 4 ]; then
    NUM_NODES=2
    TASKS_PER_NODE=2
elif [ "$PROCESSORS" -eq 9 ]; then
    NUM_NODES=3
    TASKS_PER_NODE=3
elif [ "$PROCESSORS" -eq 16 ]; then
    NUM_NODES=4
    TASKS_PER_NODE=4
elif [ "$PROCESSORS" -eq 25 ]; then
    NUM_NODES=4
    TASKS_PER_NODE=7
elif [ "$PROCESSORS" -eq 36 ]; then
    NUM_NODES=4
    TASKS_PER_NODE=9
fi

# --- Define job-specific variables ---
JOB_NAME="${PROGRAM}_p${PROCESSORS}_n${PROBLEM_SIZE}"
OUTPUT_FILE="${PROGRAM}_p${PROCESSORS}_n${PROBLEM_SIZE}_%j.out"
current_time=$(date +%s)
HOST_FILE="hosts_${current_time}.txt"

# --- Create the Slurm job script using a heredoc ---
cat <<EOF > "${PROGRAM}.job"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/uac/msc/whuang25/cmsc5702/${OUTPUT_FILE}
#SBATCH --error=/uac/msc/whuang25/cmsc5702/%x_%j.err  # Standard error log as $job_name_$job_id.err
#SBATCH --mail-user=whuang25@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00           # Wall-clock time limit (e.g., 30 minutes)

#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks=${PROCESSORS}
#SBATCH --ntasks-per-node=${TASKS_PER_NODE}
#SBATCH --cpus-per-task=1         # One CPU per task (for MPI)

# Create a hostfile based on SLURM allocated nodes
scontrol show hostnames "\$SLURM_NODELIST" | awk '{print \$0" slots=${TASKS_PER_NODE}"}' > ${HOST_FILE}

# Run the MPI program
mpiexec.openmpi --hostfile ${HOST_FILE} -n ${PROCESSORS} ./${PROGRAM} datafiles/data${PROBLEM_SIZE}

# Clean up the hostfile
rm ${HOST_FILE}
EOF

# --- Submit the generated job script ---
sbatch -p cmsc5702_hpc -q cmsc5702 "${PROGRAM}.job"
