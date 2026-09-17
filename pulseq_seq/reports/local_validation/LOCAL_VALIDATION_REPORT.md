# Local validation report

- Sequence: `/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_seq/uih790_rtspiral_realtime.seq`
- Trajectory: `/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_trajectory/fa2f521bc41db9ffa7fea48d376fafa7.mat`
- Phase tolerance: `1.0e-04 rad` (0.005730 deg)
- Trajectory direction tolerance: `0.001000 deg`

| Check | Result | Detail |
|---|---|---|
| Dimension | PASS | actual=2.0, expected=2 |
| FOV | PASS | actual=[0.36  0.32  0.006], expected=[0.36, 0.32, 0.006] m |
| SliceNumber | PASS | actual=60.0, expected=60 |
| SliceThickness | PASS | actual=0.006, expected=0.006 m |
| Matrix | PASS | actual=[240. 213.], expected=[240, 213] |
| Resolution | PASS | actual=[240. 213.], expected=[240, 213] |
| Center | PASS | actual=[120.  106.5], expected=[120.0, 106.5] |
| TE definition | PASS | actual=1.560000 ms, expected=1.560000 ms |
| TR definition | PASS | actual=7.640000 ms, expected=7.640000 ms |
| Flip angle | PASS | actual=100.0, expected=100 deg |
| ADC count | PASS | actual=21000, expected=21000 |
| Trigger count | PASS | actual=0, expected=0 |
| SLC range | PASS | actual=0..59, expected=0..59 |
| REP range | PASS | actual=0..49, expected=0..49 |
| LIN range | PASS | actual=0..6, expected=0..6 |
| Exact acquisition ordering | PASS | all acquisitions follow SLC -> REP -> LIN |
| 350 ADC per slice | PASS | expected=350 ADC for every slice |
| 7 ADC per real-time frame | PASS | expected=7 arms for each (SLC, REP) |
| ADC phase alternation | PASS | 0/pi alternation for all acquisitions; max_error=4.693e-06 rad (0.000269 deg), tolerance=1.0e-04 rad |
| RF phase alternation | PASS | adjacent RF phase differs by pi within every slice; max_error=7.346e-06 rad (0.000421 deg), tolerance=1.0e-04 rad |
| SlicePositions length | PASS | actual=60, expected=60 |
| SlicePositions spacing | PASS | mean spacing=2.000000 mm, expected=2.000000 mm |
| SlicePositions symmetry | PASS | first=-59.000 mm, last=59.000 mm |
| SlicePositions extent | PASS | centres=-59.000..59.000 mm; with 6 mm thickness -> physical support about -62..62 mm |
| trajectory field: base_k_played | PASS | present |
| trajectory field: global_arm_index | PASS | present |
| trajectory field: global_arm_angle_deg | PASS | present |
| trajectory field: sequence_signature | PASS | present |
| base_k_played shape | PASS | shape=(1260, 2), expected=[N,2] |
| global arm metadata length | PASS | actual=21000, expected=21000 |
| GA second-arm angle | PASS | actual=222.49690000 deg, expected=222.49690000 deg |
| arm 0: sample count | PASS | N=1260 |
| arm 0: endpoint direction | PASS | direction error=0.000012835 deg, tolerance=0.001000 deg |
| arm 1: sample count | PASS | N=1260 |
| arm 1: endpoint direction | PASS | direction error=0.000010281 deg, tolerance=0.001000 deg |

## Overall

**PASS**

本地检查不能替代 UIH AIDE / Virtual Scan、SAR、PNS、100° FA/B1 和 scanner interpreter 验证。
