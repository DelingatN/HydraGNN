#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J OPF2000-PE
#SBATCH -o /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf2000-pe-%j.out
#SBATCH -e /lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/job-opf2000-pe-%j.out
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 16

# Preprocess every available pglib_opf_case2000_goc group to HDF5 while
# precomputing both Laplacian and effective-resistance positional encodings.
# Runtime overrides can be supplied with, for example:
#   sbatch --export=ALL,OPF_NUM_GROUPS=4,OPF_MAX_SAMPLES=1000 <this-script>

set -eo pipefail

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/.worktrees/hetero-gps-pe-physics}
HYDRAGNN_VENV=${HYDRAGNN_VENV:-/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv}
OPF_DATA_ROOT=${OPF_DATA_ROOT:-/lustre/orion/lrn070/proj-shared/ndelingat/HydraGNN/examples/opf/dataset}
OPF_CONFIG=${OPF_CONFIG:-configs/opf_hgt_gps_pe_case2000.json}
OPF_CASE_NAME=${OPF_CASE_NAME:-pglib_opf_case2000_goc}
OPF_NUM_GROUPS=${OPF_NUM_GROUPS:-all}
OPF_MODEL_NAME=${OPF_MODEL_NAME:-OPF_HGT_GPS_PE_case2000}

source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm711.sh
source "${HYDRAGNN_VENV}/bin/activate"

export PYTHONPATH="${HYDRAGNN_ROOT}:${HYDRAGNN_VENV}/lib/python3.11/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_DIAG=1
export HYDRAGNN_DIAG_RANK=0

module unload darshan-runtime

PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${PLUGIN_PATH}/lib"
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export NCCL_NET_PLUGIN="${PLUGIN_PATH}/lib/librccl-net.so"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HSA_FORCE_FINE_GRAIN_PCIE=1

cd "${HYDRAGNN_ROOT}/examples/opf"

if [[ ! -f "${OPF_CONFIG}" ]]; then
    echo "Missing OPF config: ${PWD}/${OPF_CONFIG}" >&2
    exit 2
fi
if [[ ! -d "${OPF_DATA_ROOT}/dataset_release_1/${OPF_CASE_NAME}" ]]; then
    echo "Missing raw OPF case: ${OPF_DATA_ROOT}/dataset_release_1/${OPF_CASE_NAME}" >&2
    exit 2
fi

OUTPUT_DIR="${OPF_DATA_ROOT}/${OPF_MODEL_NAME}.h5"
if [[ -e "${OUTPUT_DIR}" && "${OPF_OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing dataset: ${OUTPUT_DIR}" >&2
    echo "Resubmit with --export=ALL,OPF_OVERWRITE=1 to replace it." >&2
    exit 2
fi

EXTRA_ARGS=()
if [[ -n "${OPF_MAX_SAMPLES:-}" ]]; then
    EXTRA_ARGS+=(--max_samples "${OPF_MAX_SAMPLES}")
fi

echo "HydraGNN root : ${HYDRAGNN_ROOT}"
echo "Input config  : ${OPF_CONFIG}"
echo "Raw case      : ${OPF_CASE_NAME}"
echo "Groups        : ${OPF_NUM_GROUPS}"
echo "Output        : ${OUTPUT_DIR}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun --export=ALL \
    -N "${SLURM_JOB_NUM_NODES}" \
    -n "$((SLURM_JOB_NUM_NODES * 8))" \
    -c 7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    python -u train_opf_solution_heterogeneous.py \
    --inputfile "${OPF_CONFIG}" \
    --data_root "${OPF_DATA_ROOT}" \
    --case_name "${OPF_CASE_NAME}" \
    --num_groups "${OPF_NUM_GROUPS}" \
    --modelname "${OPF_MODEL_NAME}" \
    --preonly \
    --hdf5 \
    "${EXTRA_ARGS[@]}"
