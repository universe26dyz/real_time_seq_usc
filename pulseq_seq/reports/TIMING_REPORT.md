# Timing report

Target: TE 0.74 ms; TR 5.67 ms; frame 40.00 ms; slice dwell about 2.00 s.

Actual final run: TE 1.560 ms; TR 7.640 ms; frame 53.480 ms; slice dwell 2.674000 s; 60-slice duration 160.440000 s.

The target is infeasible without removing required UIH RF/ADC dead time, RF and slice-selection timing, spiral readout, M1-nulled gropt rewinder, or raster constraints. target TE 0.740 ms is below minimum 1.560 ms; target TR 5.670 ms is below minimum 7.640 ms

`seq.check_timing()` passed: `True`.
