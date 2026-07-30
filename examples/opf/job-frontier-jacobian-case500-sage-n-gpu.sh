#!/bin/bash

#SBATCH -A LRN070
#SBATCH -J jac-models-500-gpu
#SBATCH -o /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf-jacobian-case500-models-gpu-%j.out
#SBATCH -e /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf-jacobian-case500-models-gpu-%j.out
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

# Add model log-folder names here. Each folder must contain config.json and a
# checkpoint named <model-name>.pk.
MODEL_NAMES=(
    "sage_g-case500"
    "heat_g-case500"
    "hgt_g-case500"
    "pna_g-case500"
    "rgat_g-case500"
    "gat_g-case500"
    "gin_g-case500"
)

DATASET_NAME=case500
MAX_SAMPLES_PER_SPLIT=100
MAX_SOURCES_PER_GRAPH=16

DATA_ROOT="$OPF_DIR/dataset"
DATASET="$DATA_ROOT/$DATASET_NAME.h5"

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate "$ENVIRONMENT_DIR"
module unload darshan-runtime || true

export PYTHONPATH="$HYDRAGNN_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

JOB_TMP="${TMPDIR:-/tmp}/opf-jacobian-${SLURM_JOB_ID}"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export XDG_CACHE_HOME="$JOB_TMP/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

for required_path in "$PYTHON_BIN" "$DATASET"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 2
    fi
done

if (( ${#MODEL_NAMES[@]} == 0 )); then
    echo "MODEL_NAMES must contain at least one model log-folder name." >&2
    exit 2
fi

# Check every model before starting any expensive analysis.
for model_name in "${MODEL_NAMES[@]}"; do
    model_dir="$OPF_DIR/logs/$model_name"
    checkpoint="$model_dir/$model_name.pk"
    config="$model_dir/config.json"
    for required_path in "$checkpoint" "$config"; do
        if [[ ! -e "$required_path" ]]; then
            echo "Missing required path: $required_path" >&2
            exit 2
        fi
    done
done

cd "$OPF_DIR"

echo "============================================================"
echo " OPF Jacobian multi-model single-GPU analysis"
echo " Models      : ${MODEL_NAMES[*]}"
echo " Dataset     : $DATASET"
echo " CPU helpers : $SLURM_CPUS_PER_TASK"
echo " GPU/task    : 1"
echo " Samples     : $MAX_SAMPLES_PER_SPLIT test samples/model"
echo " Sources     : $MAX_SOURCES_PER_GRAPH load nodes/sample"
echo "============================================================"

# Verify ROCm visibility once before processing the model array.
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

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_DIR="$OPF_DIR/logs/$MODEL_NAME"
    CHECKPOINT="$MODEL_DIR/$MODEL_NAME.pk"
    CONFIG="$MODEL_DIR/config.json"
    OUTPUT_DIR="$MODEL_DIR/jacobian_analysis_gpu_${DATASET_NAME}_${MAX_SAMPLES_PER_SPLIT}s_${MAX_SOURCES_PER_GRAPH}src"
    mkdir -p "$OUTPUT_DIR"

    echo
    echo "------------------------------------------------------------"
    echo " Model      : $MODEL_NAME"
    echo " Checkpoint : $CHECKPOINT"
    echo " Output     : $OUTPUT_DIR"
    echo "------------------------------------------------------------"

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
            --max-samples-per-split "$MAX_SAMPLES_PER_SPLIT" \
            --max-sources-per-graph "$MAX_SOURCES_PER_GRAPH" \
            --jvp-backend auto \
            --log-every 5 \
            --save-every 10 \
            --output-dir "$OUTPUT_DIR" \
            --no-plot

    echo "Completed $MODEL_NAME"
    echo "Results: $OUTPUT_DIR/radial_jacobian.json"
done

echo
echo "All model analyses complete."
