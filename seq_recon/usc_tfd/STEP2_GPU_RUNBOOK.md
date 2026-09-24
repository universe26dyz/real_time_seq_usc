# Step 2 GPU handoff runbook

Step 2A added only the offline BART port and CPU dry-run. It did not install BART locally and did not invoke `nlinv` or `pics`.

On the GPU server, first create/activate `Pulseq_gpu` with Python `numpy`, `scipy`, `ismrmrd`, and `tomli`; install a CUDA-capable BART build separately, then set its location without adding a machine-specific path to source:

```bash
conda activate Pulseq_gpu
export BART_TOOLBOX_PATH=/path/to/bart
export CUDA_VISIBLE_DEVICES=0
cd /path/to/real_time_seq_usc
python seq_recon/usc_tfd/check_bart_environment.py --bart-toolbox-path "$BART_TOOLBOX_PATH"
```

The report must be `overall: PASS`: it checks `bart`, BART's Python module, version when available, `nvidia-smi -L`, `CUDA_VISIBLE_DEVICES`, and required Python packages without changing the environment.

Then run the same no-BART dry-run against the transferred Step-1 package:

```bash
python seq_recon/usc_tfd/reconstruct_t13_slice0_tfd_bart.py \
  --prepared-npz /path/to/t13_slice0_usc_tfd_input.npz \
  --input-summary /path/to/input_summary.json \
  --config seq_recon/usc_tfd/configs/t13_5_slice0.toml \
  --out /path/to/t13_5_slice0_step2a_dry_run \
  --bart-toolbox-path "$BART_TOOLBOX_PATH" --gpu-device 0 --dry-run
```

Or use the wrapper (it always runs the environment check and the dry-run):

```bash
seq_recon/usc_tfd/scripts/run_t13_5_slice0_gpu.sh \
  /path/to/t13_slice0_usc_tfd_input.npz /path/to/input_summary.json /path/to/t13_5_slice0_step2a_dry_run
```

Inspect `tfd_dry_run_plan.json`: the expected inputs are k-space `[50,7,2,1250]`, dynamic BART k-space `[1,1250,7,2,1,1,1,1,1,1,50]`, dynamic trajectory `[3,1250,7,1,1,1,1,1,1,1,50]`, `ksp_all [1,1250,350,2]`, and `traj_all [3,1250,350]`. Their nlinv merge is USC arm-major/frame-fast, not Step-1 frame-major. Native BART axes are `[x/read,y/phase]`: solve `[x,y]=[360,360]`, center-crop native output to `[240,213]` (x `60:300`, y `73:286`), then separately reproduce USC display orientation (flip x, swap axes) to obtain `[y,x]=[213,240]`. No DCF/GIRF/B0/spatial TV is used, and raw complex output is kept before the display transform.

Only after the environment and dry-run are accepted should Step 2B explicitly enable actual BART execution. Its planned commands are USC `nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t`, followed by TFD-only `pics` with configured temporal λ `0.0002` and effective BART λ `0.01` (`0.0002 * 50`). This Step 2A runner intentionally rejects any invocation without `--dry-run`.
