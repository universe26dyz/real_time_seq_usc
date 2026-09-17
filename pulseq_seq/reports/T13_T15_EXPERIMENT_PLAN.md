# T13--T15 GA diagnostic plan

- **T13**: USC-like finite 144-entry GA LUT with a global index that wraps across slices.  It tests finite-LUT / UIH view-label compatibility without treating 144 as a temporal frame size.
- **T14**: finite 350-entry GA LUT reset at every slice.  Each slice has the same ordered 350-arm trajectory set.
- **T15**: continuous global GA with no finite trajectory reset.  LIN resets locally at each slice, while trajectory metadata preserves the global acquisition index and actual angle.

Each policy is generated with 5 slices and 60 slices.  The 5-slice case isolates trajectory and label compatibility from full-stack behaviour; the 60-slice case tests UIH GLI / overlapping-stack geometry.  For each 60-slice scanner run, set GUI-only controls to Gap 50 → Gap 0 (or the minimum allowed), Rotation OFF, FOV Shift OFF, and center the stack at isocenter.  Do not encode GUI Gap in Pulseq without an officially supported UIH Definition.
