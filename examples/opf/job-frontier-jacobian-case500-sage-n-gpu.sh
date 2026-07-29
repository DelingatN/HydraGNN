#!/bin/bash

#SBATCH -A LRN070
#SBATCH -J jac-sage-n-500-gpu
#SBATCH -o /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf-jacobian-case500-sage-n-gpu-%j.out
#SBATCH -e /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf-jacobian-case500-sage-n-gpu-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 7

set -euo pipefail

HYDRAGNN_ROOT=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN
OPF_DIR="$HYDRAGNN_ROOT/examples/opf"
PYTHON_BIN="$HYDRAGNN_ROOT/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/bin/python3.11"
ENVIRONMENT_DIR="$HYDRAGNN_ROOT/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv"

MODEL_NAME=sage_n-case500
DATASET_NAME=case500
CHECKPOINT="$OPF_DIR/logs/$MODEL_NAME/$MODEL_NAME.pk"
CONFIG="$OPF_DIR/logs/$MODEL_NAME/config.json"
DATA_ROOT="$OPF_DIR/dataset"
DATASET="$DATA_ROOT/$DATASET_NAME.h5"
OUTPUT_DIR="$OPF_DIR/logs/$MODEL_NAME/jacobian_analysis_gpu_case500"

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "$ENVIRONMENT_DIR"
module unload darshan-runtime || true

export PYTHONPATH="$HYDRAGNN_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

JOB_TMP="${TMPDIR:-/tmp}/opf-jacobian-${SLURM_JOB_ID}"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export XDG_CACHE_HOME="$JOB_TMP/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUTPUT_DIR"

for required_path in "$PYTHON_BIN" "$CHECKPOINT" "$CONFIG" "$DATASET"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 2
    fi
done

cd "$OPF_DIR"

echo "============================================================"
echo " OPF Jacobian single-GPU timing benchmark"
echo " Model       : $MODEL_NAME"
echo " Dataset     : $DATASET"
echo " Checkpoint  : $CHECKPOINT"
echo " CPU helpers : $SLURM_CPUS_PER_TASK"
echo " GPU/task    : 1"
echo " Samples     : 10 test samples"
echo " Sources     : 4 load nodes/sample"
echo " Directions  : 80 total (4 loads x Pd/Qd x 10 samples)"
echo " Output      : $OUTPUT_DIR"
echo "============================================================"

srun \
    --kill-on-bad-exit=1 \
    -N1 \
    -n1 \
    -c"$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    "$PYTHON_BIN" -u -c \
        "import torch; assert torch.cuda.is_available(), 'ROCm GPU is not visible'; print('torch=', torch.__version__, 'hip=', torch.version.hip, 'device=', torch.cuda.get_device_name(0))"

srun \
    --kill-on-bad-exit=1 \
    -N1 \
    -n1 \
    -c"$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    "$PYTHON_BIN" -u analyze_opf_jacobian.py \
        --modelname "$MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --data-root "$DATA_ROOT" \
        --format hdf5 \
        --device auto \
        --splits test \
        --max-samples-per-split 10 \
        --max-sources-per-graph 4 \
        --jvp-backend auto \
        --log-every 1 \
        --save-every 0 \
        --output-dir "$OUTPUT_DIR" \
        --no-plot

echo
echo "Benchmark complete."
echo "Results: $OUTPUT_DIR/radial_jacobian.json"
