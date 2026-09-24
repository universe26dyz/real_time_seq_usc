# USC TFD offline preparation

This directory is limited to Step 1: an offline, CPU-safe adapter that validates one UIH H5 slice against the matching external Pulseq MAT and prepares arrays for a later BART TFD run. It does not reconstruct an image.

The default configuration uses `pair_mean` only as an explicit compatibility choice with the existing QC baseline. The 2520:1260 integer ratio does not prove that pair averaging is the scanner's intended oversampling-removal operation.

```bash
conda run --no-capture-output -n Pulseq python seq_recon/usc_tfd/prepare_t13_slice0.py \
  --h5 /media/universe/DATA/lab/SVR/data/20260917_real_time_seq_usc/h5/UID_7685991886275844011_pulseq_T13_5/testdata.h5 \
  --traj /home/universe/SVR/real_time_seq_usc/pulseq_seq/diagnostic_sequences/T13_5slice_usc144_lut/out_trajectory/b83a6a3dd57599fa51d5a6134d4fd590.mat \
  --slice 0 --config seq_recon/usc_tfd/configs/t13_5_slice0.toml \
  --out /tmp/t13_5_slice0_usc_tfd_prepare
```

The resulting NPZ contains `[frame, arm, coil, sample]` k-space and `[frame, arm, sample, 2]` BART-convention trajectory. BART trajectory coordinates use `[kx, -ky]`, measured-trajectory `kmax` normalization, and scaling to the 360-square working matrix. The future Step 2 must produce the documented central `[y, x] = [213, 240]` crop after working-space reconstruction.
