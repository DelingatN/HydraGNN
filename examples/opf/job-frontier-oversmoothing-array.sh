#!/bin/bash

#SBATCH -A LRN070
#SBATCH -J opf-oversmoothing
#SBATCH -o /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf-oversmoothing-%A_%a.out
#SBATCH -e /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf-oversmoothing-%A_%a.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --array=0-20
##SBATCH -C nvme
##SBATCH -S 1

set -euo pipefail

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

HYDRAGNN_ROOT=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source activate /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$HYDRAGNN_ROOT:$PYTHONPATH"
export PYTHONPATH="/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/lib/python3.11/site-packages/:$PYTHONPATH"

PYTHON_BIN=/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv/bin/python3.11

which "$PYTHON_BIN"
"$PYTHON_BIN" -c "import adios2; print(adios2.__version__, adios2.__file__)"
"$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.__file__)"

module unload darshan-runtime || true
module list

echo "$LD_LIBRARY_PATH" | tr ':' '\n'

module load rccl-net-plugin/1.0
export PLUGIN_PATH=$OLCF_OFI_NCCL_ROOT
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PLUGIN_PATH}/lib"

export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1

export NCCL_NET_PLUGIN="${PLUGIN_PATH}/lib/librccl-net.so"
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"

unset NCCL_NET_PLUGIN
unset NCCL_NET

export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0

env | grep ROCM || true
env | grep ^MI || true
env | grep ^MPICH || true
env | grep ^HYDRA || true

cd "$HYDRAGNN_ROOT/examples/opf"

which "$PYTHON_BIN"
"$PYTHON_BIN" -c "import numpy; print(numpy.__version__)"

CONFIG_FILES=(
    # "oversmoothing_configs/01_heterogin_no_gps.json"
    # "oversmoothing_configs/02_heterogin_gps.json"
    # "oversmoothing_configs/03_heterosage_no_gps.json"
    # "oversmoothing_configs/04_heterosage_gps.json"
    # "oversmoothing_configs/05_heterogat_no_gps.json"
    # "oversmoothing_configs/06_heterogat_gps.json"
    # "oversmoothing_configs/07_heteropna_no_gps.json"
    # "oversmoothing_configs/08_heteropna_gps.json"
    # "oversmoothing_configs/09_heterorgat_no_gps.json"
    # "oversmoothing_configs/10_heterorgat_gps.json"
    # "oversmoothing_configs/11_heterohgt_no_gps.json"
    # "oversmoothing_configs/12_heterohgt_gps.json"
    # "oversmoothing_configs/13_heteroheat_no_gps.json"
    # "oversmoothing_configs/14_heteroheat_gps.json"
    "oversmoothing_configs/15_heterosage_attention_only_gps.json"
    "oversmoothing_configs/16_heteroheat_depth_01_no_gps.json"
    # "oversmoothing_configs/17_heteroheat_depth_02_no_gps.json"
    # "oversmoothing_configs/18_heteroheat_depth_04_no_gps.json"
    # "oversmoothing_configs/19_heteroheat_depth_08_no_gps.json"
    # "oversmoothing_configs/20_heteroheat_depth_12_no_gps.json"
    "oversmoothing_configs/21_heteroheat_depth_16_no_gps.json"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch.}
if ((TASK_ID < 0 || TASK_ID >= ${#CONFIG_FILES[@]})); then
    echo "Error: array task $TASK_ID is outside 0-$((${#CONFIG_FILES[@]} - 1))."
    exit 2
fi

RUN_CONFIG=${CONFIG_FILES[$TASK_ID]}
if [[ ! -f "$RUN_CONFIG" ]]; then
    echo "Error: configuration does not exist: $RUN_CONFIG"
    exit 2
fi

CASE_NAME=${CASE_NAME:-pglib_opf_case14_ieee}
MODEL_NAME=${MODEL_NAME:-case14}
NUM_GROUPS=${NUM_GROUPS:-1}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-32}

CONFIG_NAME=$(basename "$RUN_CONFIG" .json)
RUN_NAME=${RUN_NAME:-"${CONFIG_NAME}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"}

echo
echo "============================================================"
echo " HydraGNN oversmoothing array training"
echo " Array task  : $TASK_ID / $((${#CONFIG_FILES[@]} - 1))"
echo " Config      : $RUN_CONFIG"
echo " Run name    : $RUN_NAME"
echo " Case        : $CASE_NAME"
echo " Nodes       : $SLURM_JOB_NUM_NODES"
echo " GPUs/ranks  : $((SLURM_JOB_NUM_NODES * 8))"
echo " Epochs      : $EPOCHS"
echo " Batch size  : $BATCH_SIZE"
echo "============================================================"
echo

srun \
    --kill-on-bad-exit=1 \
    --export=ALL,HYDRAGNN_DIAG=1,HYDRAGNN_DIAG_RANK=0 \
    -N"$SLURM_JOB_NUM_NODES" \
    -n"$((SLURM_JOB_NUM_NODES * 8))" \
    --ntasks-per-node=8 \
    -c7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    "$PYTHON_BIN" -u train_opf_solution_heterogeneous.py \
        --hdf5 \
        --inputfile "$RUN_CONFIG" \
        --case_name "$CASE_NAME" \
        --num_groups "$NUM_GROUPS" \
        --num_epoch "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --modelname "$MODEL_NAME" \
        --log "$RUN_NAME" \
        --node_target_type bus
