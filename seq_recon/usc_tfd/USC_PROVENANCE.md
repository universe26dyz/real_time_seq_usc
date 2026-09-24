# USC provenance and UIH adapter boundary

Reference repository: <https://github.com/usc-mrel/python-ismrmrd-server>  
Pinned reference commit: `faaf0f3e44bf8f2557ac4d7da90a62a5451d7b96`  
Primary source: `rtspiral_bart_tvrecon.py`
Supporting source inspected: `reconutils.py`

## BART compatibility pin

The pinned USC Dockerfile builds BART `v0.7.00`, but it is stale relative to the pinned current `rtspiral_bart_tvrecon.py`: the latter uses `nufft -g -x <Nx>:<Ny>:1 -a` for scale estimation. Official BART v0.7.00 and v0.8.00 expose the corresponding image-dimension option as `-d`; official BART v0.9.00 exposes `-x` and retains `-d` only as deprecated compatibility syntax.

This project therefore pins BART `v0.9.00` (tag commit `672a840ff88117e09dc9803e0d9c8c5a7f1c42a9`) for the first formal experiment. This is a project compatibility pin derived from source-level CLI matching; it is not a claim that USC authors explicitly selected v0.9.00. The audit also verified that v0.9.00 supports the formal `nlinv` options, the `pics` options and generalized `-R T:A:B:C` syntax, and the `import bart; bart.bart(nargout, command, ...)` API with `BART_TOOLBOX_PATH` and legacy `TOOLBOX_PATH` support.

## USC blocks ported in Step 2A

`bart_tfd.py` is an offline port of these blocks in `rtspiral_bart_tvrecon.py`:

- `data = np.array(data) * 1e3`, dynamic `[sample, arm, coil, time]` BART layout, zero-kz trajectory construction, and 11-dimensional `pics` singleton layout;
- all-frame `ksp_all` and `traj_all` preparation for `nlinv`, including the source's arm-major/frame-fast merged acquisition dimension (`arm 0: frame 0..49`, then arm 1);
- exact `nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t` call and the IFFT → 720 resize → FFT → 360 resize → `normalize 8` sensitivity-map path;
- `estimate_scale_bart()`'s adjoint-NUFFT command and median/p90/max conditional;
- temporal `pics` command construction, including `reg_lambda_temporal * n_frame_per_chunk` and optional `-R W:3:0:<lambda>` branch;
- USC's three-frame chunk overlap and result trimming.

`reconutils.py` supplies the original streaming configuration/trajectory/GIRF context. Its MRD-streaming and trajectory-loading functions are deliberately not copied: Step 1 supplies a validated offline NPZ instead.

## Project-specific adaptation boundary

- MRD streaming becomes a prepared offline NPZ; no H5/MAT parsing occurs in the BART core.
- UIH 2520-to-1260 `pair_mean` ADC adaptation and nominal pre-discard occur upstream in Step 1.
- The trajectory is the Step-1 `[kx,-ky]` trajectory, already scaled to the 360 working matrix; no GIRF or B0 correction is added.
- Reconstruction is fixed at 50 offline frames rather than an incoming runtime stream. Step-1 remains frame-major, but the `nlinv` merge reproduces USC's arm-major/frame-fast order from the dynamic arrays.
- Future native complex arrays are saved to disk with BART spatial axes `[x/read, y/phase, time]`, not converted to MRD images. USC display orientation is a separate flip of native x followed by an x/y swap.
- The native centered crop is `[x,y] [360,360] -> [240,213]` (x `60:300`, y `73:286`); the separate USC display transform produces `[y,x] [213,240]`. Neither operation modifies BART encoding.

The formal USC BART path does not use the baseline Hoge DCF, RSS, ESPIRiT, SigPy, spatial regularization, GIRF, or B0 correction. Step 2A implements this path but only dry-runs it; no BART command has been executed.
