# Step 2 GPU handoff runbook (do not execute during Step 1)

Step 1 created a CPU-validated input package; it did not install BART or run a reconstruction. On the GPU server, use the `Pulseq_gpu` environment and verify these prerequisites before adding or invoking the USC BART solver:

```bash
conda activate Pulseq_gpu
command -v bart
nvidia-smi
python -c "import ismrmrd, numpy, scipy; print('Python input dependencies available')"
```

Use the generated `t13_slice0_usc_tfd_input.npz` only after checking that its summary states:

- k-space: `[50, 7, 2, 1250]` (`frame, arm, coil, sample`);
- BART trajectory: `[50, 7, 1250, 2]`, convention `[kx, -ky]`, normalized by its recorded measured `kmax` and scaled to 360/2;
- `pair_mean` is an explicit compatibility selection, not a proven physical oversampling interpretation;
- H5/MAT slice, LIN, local-arm, and REP checks are all true.

The planned USC reference is `rtspiral_bart_tvrecon.py` at commit `faaf0f3e44bf8f2557ac4d7da90a62a5451d7b96`: first obtain coil maps with BART `nlinv`, then perform BART `pics` with the configured temporal finite difference term (`reg_lambda_temporal=0.0002`, no spatial TV, 120 iterations, one chunk). Preserve the USC BART regularization scaling; do not reinterpret lambda. Work in the 360-square space and only then create the documented central `[y, x] = [213, 240]` crop.

This runbook intentionally contains no executable BART command: implementing or running the optimizer is Step 2 and is outside the validated CPU-only Step 1 boundary.
