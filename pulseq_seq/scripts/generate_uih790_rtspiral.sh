#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-uih-rtspiral}"

cd "${PROJECT_ROOT}"
conda run --no-capture-output -n Pulseq python src/write_rtspiral_svr_uih.py \
  --config config/uih790_rtspiral_realtime.json \
  --output-root "${PROJECT_ROOT}"
