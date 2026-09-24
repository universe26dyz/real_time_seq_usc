# USC provenance and UIH adapter boundary

Reference repository: <https://github.com/usc-mrel/python-ismrmrd-server>  
Pinned reference commit: `faaf0f3e44bf8f2557ac4d7da90a62a5451d7b96`  
Primary source: `rtspiral_bart_tvrecon.py`
Supporting source inspected: `reconutils.py`

## USC blocks ported in Step 2A

`bart_tfd.py` is an offline port of these blocks in `rtspiral_bart_tvrecon.py`:

- `data = np.array(data) * 1e3`, dynamic `[sample, arm, coil, time]` BART layout, zero-kz trajectory construction, and 11-dimensional `pics` singleton layout;
- all-frame `ksp_all` and `traj_all` preparation for `nlinv`;
- exact `nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t` call and the IFFT → 720 resize → FFT → 360 resize → `normalize 8` sensitivity-map path;
- `estimate_scale_bart()`'s adjoint-NUFFT command and median/p90/max conditional;
- temporal `pics` command construction, including `reg_lambda_temporal * n_frame_per_chunk` and optional `-R W:3:0:<lambda>` branch;
- USC's three-frame chunk overlap and result trimming.

`reconutils.py` supplies the original streaming configuration/trajectory/GIRF context. Its MRD-streaming and trajectory-loading functions are deliberately not copied: Step 1 supplies a validated offline NPZ instead.

## Project-specific adaptation boundary

- MRD streaming becomes a prepared offline NPZ; no H5/MAT parsing occurs in the BART core.
- UIH 2520-to-1260 `pair_mean` ADC adaptation and nominal pre-discard occur upstream in Step 1.
- The trajectory is the Step-1 `[kx,-ky]` trajectory, already scaled to the 360 working matrix; no GIRF or B0 correction is added.
- Reconstruction is fixed at 50 offline frames rather than an incoming runtime stream. The all-frame `nlinv` arrays retain Step-1 frame-major acquisition order.
- Future native complex arrays are saved to disk, not converted to MRD images. Any USC display flip/transpose is intentionally separate from the native BART orientation.
- A deterministic centered `[360,360] -> [213,240]` crop is applied only after native solving. It does not modify BART encoding.

The formal USC BART path does not use the baseline Hoge DCF, RSS, ESPIRiT, SigPy, spatial regularization, GIRF, or B0 correction. Step 2A implements this path but only dry-runs it; no BART command has been executed.
