#!/bin/bash
#SBATCH --job-name=wake_sleep_ddp
#SBATCH -p sched_mit_psfc_gpu_r8      # queue / partition
#SBATCH -N 4                        # 4 nodes
#SBATCH --ntasks-per-node=1           # one task per node (torchrun will fan-out the GPUs)
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=256GB
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out

# ──────────────────────────────────────────────────────────────
# 1.  Activate environment
# ──────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
module load deprecated-modules gcc/12.2.0-x86_64
module load python/3.10.8-x86_64        # provides “python” + “torchrun”
# If you normally `conda activate …`, do it here
# conda activate torch

cd /pool001/spangher/wake-sleep-style-classifier || exit 1

# ──────────────────────────────────────────────────────────────
# 2.  Pick a master & discover its IP
# ──────────────────────────────────────────────────────────────
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
master_node=${nodes[0]}
master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)

echo "============================================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $master_node   (rank 0 / $SLURM_NNODES)"
echo "Master   : $master_ip:1234"
echo "Start    : $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# Enable CUDA_LAUNCH_BLOCKING for debugging CUDA errors
# This makes CUDA operations synchronous, which helps identify
# the exact location of CUDA errors in the code
# ──────────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=1

# ──────────────────────────────────────────────────────────────
# 3.  Launch one torchrun per node, one process per GPU
# ──────────────────────────────────────────────────────────────
srun /home/software/anaconda3/2023.07/bin/torchrun \
     --nnodes="$SLURM_NNODES" \
     --nproc-per-node="$SLURM_GPUS_PER_NODE" \
     --rdzv-id="$SLURM_JOB_ID" \
     --rdzv-backend=c10d \
     --rdzv-endpoint="$master_ip:1234" \
     wake_pl.py \
         --devices 4 \
         --num_nodes "$SLURM_NNODES" \
         --data_dir "datasets/aria-midi-v1-deduped-ext/data" \
         --strategy ddp_find_unused_parameters_true          # remove if wake_pl.py sets it automatically
