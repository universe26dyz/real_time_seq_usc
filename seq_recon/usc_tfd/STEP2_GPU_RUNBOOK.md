# Step 2 GPU handoff runbook

This procedure is for the GPU server only. Step 2B-1 must stop after environment validation; do not run real T13 `nlinv` or `pics`.

## 1. Environment setup

Keep BART outside the repository and use the USC Dockerfile target: BART `v0.7.00`, tag commit `d1b0e576c3f759089915565d5bf57832acf7b03e`. The source Makefile defaults to `CUDA=0`; its supported CUDA build is `make CUDA=1 CUDA_BASE=<CUDA prefix>`. Before building, confirm that `nvcc`, its matching CUDA libraries, FFTW, LAPACKE, PNG, OpenBLAS, and a C/C++ compiler are already available. Do not use `sudo` or alter this repository.

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
git clone --branch v0.7.00 --depth 1 https://github.com/mrirecon/bart.git "$HOME/software/bart-v0.7.00"
git -C "$HOME/software/bart-v0.7.00" rev-parse HEAD  # expect d1b0e576c3f759089915565d5bf57832acf7b03e
make -C "$HOME/software/bart-v0.7.00" CUDA=1 CUDA_BASE="$CUDA_PREFIX" -j"$(nproc)"
```

If the import probe reports a missing package, install only that package into `Pulseq_gpu`; do not recreate the environment. Do not use `make install`: the user-owned source directory itself is the BART toolbox.

## 2. Environment validation

Set both BART variables. `BART_TOOLBOX_PATH` is this project’s configuration name; BART v0.7.00 `python/bart.py` additionally requires `TOOLBOX_PATH` for `bart.bart(...)`.

```bash
conda activate Pulseq_gpu
export BART_TOOLBOX_PATH="$HOME/software/bart-v0.7.00"
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

The smoke test uses synthetic arrays only: BART Python NumPy round-trip plus `nufft -g -a -d 8:8:1`. It never reads T13 data. The environment report records host, Python/conda details, BART executable/version/Python API, GPU/driver, CUDA compiler, required commands, package versions, and smoke result.

### Compatibility gate — currently blocking Step 2B-2

The formal pinned USC scale command is `nufft -g -x <Nx>:<Ny>:1 -a`. Official BART v0.7.00 source exposes `nufft -d x:y:z`, not `-x`; the new checker tests this exact option and must remain `FAIL` if `-x` is absent. Do **not** silently replace the formal USC `-x` with `-d` or change BART versions. Capture the server’s `bart nufft -h` output and resolve the source/version discrepancy before real reconstruction.

## 3. Dry-run

Only after the compatibility gate is resolved, run the no-BART dry-run against the transferred Step-1 package:

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

Only after environment, smoke test, compatibility gate, and dry-run are all accepted should Step 2B-2 explicitly enable real BART execution. Its planned commands remain USC `nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t`, followed by TFD-only `pics` with configured temporal λ `0.0002` and effective BART λ `0.01` (`0.0002 * 50`). The current runner intentionally rejects any invocation without `--dry-run`.
