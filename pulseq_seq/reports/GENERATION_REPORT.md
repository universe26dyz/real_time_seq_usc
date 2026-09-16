# UIH RTSpiral final generation

- USC RTSpiral: https://github.com/usc-mrel/rtspiral_pypulseq @ `5d32a138e84ce3df1303c4a98d52739611e6ec9c`
- PulseqSystems: https://github.com/nimpulseq/PulseqSystems @ `d85ba90c473b6d7eca49806e10545df427965e41`
- Python / PyPulseq: 3.10.19 / 1.4.2.post1
- UIH adaptations: JSON config, UIH timing, fixed-slice 7×50×60 acquisition, SLC/REP/LIN, UIH Definitions, and traceable trajectory metadata.
- FOV: user-specified 360×320 mm; radial trajectory design FOV 360 mm is an engineering derivation, not a paper parameter.
- Matrix / Center / Resolution definition: [240, 213] / [120.0, 106.5] / [240, 213].
- Actual TE/TR: 1.560/7.640 ms; frame 53.480 ms; slice dwell 2.674000 s; total 160.440000 s.
- Max physical-axis gradient: 14.849998 mT/m.
- Rewinder: requested 3 ms; solver bound 4.0 ms; actual 2.780 ms.
- Signature: `fa2f521bc41db9ffa7fea48d376fafa7`; timing check: `True`.
- Sequence: `/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_seq/uih790_rtspiral_realtime.seq` (238730593 bytes)
- Trajectory: `/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_trajectory/fa2f521bc41db9ffa7fea48d376fafa7.mat`

Next: import the `.seq` into UIH AIDE / Pulseq Virtual Scan, inspect arbitrary gradients and definitions, then validate SAR, PNS, 100° FA/B1, label-to-DHL mapping, and raw-data timestamps.
