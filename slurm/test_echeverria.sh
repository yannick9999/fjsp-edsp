#!/bin/bash
#SBATCH --job-name=fjsp_test_echeverria
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_echeverria_%j.out
#SBATCH --error=logs/test_echeverria_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

MODEL_DIR="./models_echeverria"

# Echeverria's pretrained policies must exist (saved as .pth, not .pt)
if ! ls "${MODEL_DIR}"/*.pth 1>/dev/null 2>&1; then
    echo "ERROR: No .pth file found in ${MODEL_DIR}" >&2
    exit 1
fi

echo "=== Testing Echeverria pretrained policies model_dir=${MODEL_DIR} ==="
srun nvidia-smi || true

exec srun "${PYTHON}" run_test_suite_echeverria.py
