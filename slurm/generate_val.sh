#!/bin/bash
#SBATCH --job-name=fjsp_val
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/val_seed%a_%j.out
#SBATCH --error=logs/val_seed%a_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"
mkdir -p logs

# no cuda module, CP-SAT runs on CPU and must stay out of the torch process
module purge
module load StdEnv/2023 python/3.11.5

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

SEED=${SLURM_ARRAY_TASK_ID}
echo "=== Generating validation set seed=${SEED} ==="

exec "${PYTHON}" generate_val.py --seed "${SEED}"
