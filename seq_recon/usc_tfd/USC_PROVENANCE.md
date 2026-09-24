# USC provenance and UIH adapter boundary

Reference repository: <https://github.com/usc-mrel/python-ismrmrd-server>  
Pinned reference commit: `faaf0f3e44bf8f2557ac4d7da90a62a5451d7b96`  
Primary Step 2 source: `rtspiral_bart_tvrecon.py`  
Supporting source: `reconutils.py`

## USC logic retained for the later Step 2

- temporal frame organization driven by a reconstruction choice (`arms_per_frame`), not scanner REP;
- trajectory convention `[kx, -ky]` and `kmax`-based scaling into the working reconstruction matrix;
- square FOV-oversampled working space before a final crop;
- the planned BART `nlinv` sensitivity-map and `pics` temporal finite-difference workflow, including its regularization scaling and chunking configuration.

## Project-specific Step 1 adapter

- direct offline ISMRMRD H5 reading rather than the USC client/server connection;
- select `idx.slice == 0` while preserving original H5 acquisition ordinals;
- validate T13 H5 `LIN` and slice ordering against `trajectory_index_per_acq` and `slice_index_per_acq` from the external MAT;
- apply an explicitly selected UIH 2520-to-1260 ADC reduction (`pair_mean`, `even`, or `odd`), then discard 10 nominal samples;
- use `base_k_played`, `global_arm_angle_deg`, and the MAT's actual per-acquisition metadata;
- carry the measured H5 geometry forward without replacing it with the original 2-mm design intent.

No USC source code is copied in Step 1. BART calls, DCF use, coil-map estimation, GIRF, B0 correction, spatial TV, and TFD optimization are intentionally deferred to Step 2.
