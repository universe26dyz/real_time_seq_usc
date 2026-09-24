# Step 2 GPU handoff runbook

This procedure is for the GPU server only. Step 2B-1 must stop after environment validation; do not run real T13 `nlinv` or `pics`.

The server-side environment was manually validated on `CMRServer04`: RTX 3090, CUDA 11.8, BART v0.9.00 commit `672a840ff88117e09dc9803e0d9c8c5a7f1c42a9`, and synthetic `nufft -g -x 8:8:1 -a` PASS. This is an environment record only; no real T13 reconstruction result is claimed here.

## 1. Environment setup

Keep BART outside the repository and use the project compatibility pin BART `v0.9.00`, tag commit `672a840ff88117e09dc9803e0d9c8c5a7f1c42a9`. This supersedes the old USC Dockerfile's v0.7.00 only because the pinned current USC TFD source requires `nufft -x`; it is a source-level CLI compatibility decision, not an attribution to USC authors. The v0.9.00 Makefile defaults to `CUDA=0`; its supported CUDA build is `make CUDA=1 CUDA_BASE=<CUDA prefix>`. Before building, confirm that `nvcc`, its matching CUDA libraries, FFTW, LAPACKE, PNG, OpenBLAS, and a C/C++ compiler are already available. Do not use `sudo` or alter this repository.

```bash
conda activate Pulseq_gpu
python - <<'PY'
for name in ("numpy", "scipy", "matplotlib", "tomli", "ismrmrd"):
    __import__(name)
    print(f"{name}: available")
PY

nvcc --version
CUDA_PREFIX="$(dirname "$(dirname "$(command -v nvcc)")")"
mkdir -p "$HOME/software"
git clone --branch v0.9.00 --depth 1 https://github.com/mrirecon/bart.git "$HOME/software/bart-v0.9.00"
git -C "$HOME/software/bart-v0.9.00" rev-parse HEAD  # expect 672a840ff88117e09dc9803e0d9c8c5a7f1c42a9
make -C "$HOME/software/bart-v0.9.00" CUDA=1 CUDA_BASE="$CUDA_PREFIX" -j"$(nproc)"
```

If the import probe reports a missing package, install only that package into `Pulseq_gpu`; do not recreate the environment. Do not use `make install`: the user-owned source directory itself is the BART toolbox.

## 2. Environment validation

Set `BART_TOOLBOX_PATH`, which BART v0.9.00 `python/bart.py` recognizes directly. Exporting `TOOLBOX_PATH` as well preserves its supported legacy fallback.

```bash
conda activate Pulseq_gpu
export BART_TOOLBOX_PATH="$HOME/software/bart-v0.9.00"
export TOOLBOX_PATH="$BART_TOOLBOX_PATH"
export PATH="$BART_TOOLBOX_PATH:$PATH"
export PYTHONPATH="$BART_TOOLBOX_PATH/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
cd /path/to/real_time_seq_usc
python seq_recon/usc_tfd/smoke_test_bart_gpu.py \
  --bart-toolbox-path "$BART_TOOLBOX_PATH" --out seq_recon/usc_tfd/outputs/step2b_bart_gpu_smoke.json
python seq_recon/usc_tfd/check_bart_environment.py \
  --bart-toolbox-path "$BART_TOOLBOX_PATH" --gpu-smoke-report seq_recon/usc_tfd/outputs/step2b_bart_gpu_smoke.json \
  --out seq_recon/usc_tfd/outputs/step2b_environment_report.json
```

The smoke test uses synthetic arrays only: BART Python NumPy round-trip plus `nufft -g -x 8:8:1 -a`. It never reads T13 data. The environment report records host, Python/conda details, BART executable/version/Python API, GPU/driver, CUDA compiler, required commands/options, package versions, and smoke result.

### Compatibility gate

The formal pinned USC scale command is `nufft -g -x <Nx>:<Ny>:1 -a`. BART v0.9.00 source exposes `-x x:y:z` (with legacy `-d` deprecated), matching the formal command. The checker requires exact v0.9.00 identity and audits `nufft -g/-x/-a`, all required `nlinv` options, and all required `pics` options including `-R`; do **not** silently accept another BART release or translate USC `-x` to `-d`.

## 3. Dry-run

Only after the v0.9.00 compatibility gate and synthetic smoke test pass, run the no-BART dry-run against the transferred Step-1 package:

```bash
python seq_recon/usc_tfd/reconstruct_t13_slice0_tfd_bart.py \
  --prepared-npz /path/to/t13_slice0_usc_tfd_input.npz \
  --input-summary /path/to/input_summary.json \
  --config seq_recon/usc_tfd/configs/t13_5_slice0.toml \
  --out /path/to/t13_5_slice0_step2a_dry_run \
  --bart-toolbox-path "$BART_TOOLBOX_PATH" --gpu-device 0 --dry-run
```

Or use the wrapper (it exports `TOOLBOX_PATH`, BART paths, 32-thread OpenBLAS/OMP defaults, and configurable `GPU_DEVICE`; it still performs only the dry-run):

```bash
seq_recon/usc_tfd/scripts/run_t13_5_slice0_gpu.sh \
  /path/to/t13_slice0_usc_tfd_input.npz /path/to/input_summary.json /path/to/t13_5_slice0_step2a_dry_run
```

Inspect `tfd_dry_run_plan.json`: the expected inputs are k-space `[50,7,2,1250]`, dynamic BART k-space `[1,1250,7,2,1,1,1,1,1,1,50]`, dynamic trajectory `[3,1250,7,1,1,1,1,1,1,1,50]`, `ksp_all [1,1250,350,2]`, and `traj_all [3,1250,350]`. Their nlinv merge is USC arm-major/frame-fast, not Step-1 frame-major. Native BART axes are `[x/read,y/phase]`: solve `[x,y]=[360,360]`, center-crop native output to `[240,213]` (x `60:300`, y `73:286`), then separately reproduce USC display orientation (flip x, swap axes) to obtain `[y,x]=[213,240]`. No DCF/GIRF/B0/spatial TV is used, and raw complex output is kept before the display transform.

## 4. Actual reconstruction (next step only)

Only after environment, smoke test, compatibility gate, and dry-run are all accepted should Step 2B-2 explicitly enable real BART execution. The runner requires deliberate `--execute` and performs USC `nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t`, source-locked scale `nufft -g -x 360:360:1 -a` on frames `5:50`, then TFD-only `pics` with configured temporal λ `0.0002` and effective BART λ `0.01` (`0.0002 * 50`). The selected scale is passed directly to `pics -w`, without a reciprocal.

```bash
conda activate Pulseq_gpu
export BART_TOOLBOX_PATH="$HOME/software/bart-v0.9.00"
export TOOLBOX_PATH="$BART_TOOLBOX_PATH"
export PATH="$BART_TOOLBOX_PATH:$PATH"
export PYTHONPATH="$BART_TOOLBOX_PATH/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32
cd /data/dengyz/code/real_time_seq_usc
python seq_recon/usc_tfd/reconstruct_t13_slice0_tfd_bart.py \
  --prepared-npz seq_recon/usc_tfd/outputs/T13_5_slice0_pair_mean/t13_slice0_usc_tfd_input.npz \
  --input-summary seq_recon/usc_tfd/outputs/T13_5_slice0_pair_mean/input_summary.json \
  --config seq_recon/usc_tfd/configs/t13_5_slice0.toml \
  --out seq_recon/usc_tfd/outputs/T13_5_slice0_pair_mean/tfd_bart_real \
  --bart-toolbox-path "$BART_TOOLBOX_PATH" --gpu-device 0 --execute
```

This command is documented for manual server use only; do not run it on the local CPU host.
