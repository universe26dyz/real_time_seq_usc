#!/usr/bin/env bash
set -euo pipefail

: "${BART_TOOLBOX_PATH:?Set BART_TOOLBOX_PATH to the BART installation on the GPU server.}"
: "${PULSEQ_GPU_PYTHON:=python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPARED_NPZ="${1:?first argument: Step-1 t13_slice0_usc_tfd_input.npz}"
INPUT_SUMMARY="${2:?second argument: Step-1 input_summary.json}"
OUT_DIR="${3:?third argument: output directory}"
CONFIG_PATH="${4:-${SCRIPT_DIR}/configs/t13_5_slice0.toml}"
GPU_DEVICE="${GPU_DEVICE:-0}"

export TOOLBOX_PATH="${BART_TOOLBOX_PATH}"
export PATH="${BART_TOOLBOX_PATH}:${PATH}"
export PYTHONPATH="${BART_TOOLBOX_PATH}/python${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"

"${PULSEQ_GPU_PYTHON}" "${SCRIPT_DIR}/check_bart_environment.py" --bart-toolbox-path "${BART_TOOLBOX_PATH}"
"${PULSEQ_GPU_PYTHON}" "${SCRIPT_DIR}/reconstruct_t13_slice0_tfd_bart.py" \
  --prepared-npz "${PREPARED_NPZ}" --input-summary "${INPUT_SUMMARY}" --config "${CONFIG_PATH}" \
  --out "${OUT_DIR}" --bart-toolbox-path "${BART_TOOLBOX_PATH}" --gpu-device "${GPU_DEVICE}" --dry-run
