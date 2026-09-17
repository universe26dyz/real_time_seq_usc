# UIH uMR 790 real-time spiral bSSFP

This project adapts USC RTSpiral to a UIH uMR 790 fixed-slice, 2D multi-slice real-time trueFISP/bSSFP acquisition. It retains the USC VDS spiral, M1-nulled `gropt` rewinder, and golden-angle rotation with UIH timing, no triggers, physical-support FOV-z, and output provenance.

Scanner acquisition uses 350 continuous arms per fixed slice. The downstream reconstruction default groups 7 consecutive arms into one temporal frame (50 derived frames per slice); it is not written as a scanner label. Scanner labels are `SLC` (slice index) and `LIN` (trajectory/view or local-arm index by GA policy); no `REP` temporal-frame label is used.

The verified diagnostic policies are T13 `usc144_lut_global_wrap`, T14 `lut350_slice_reset`, and T15 `continuous_global_ga`, each tested on 5- and 60-slice stacks. The six scanner runs completed successfully. For the former 60-slice UIH warning, the observed resolution was manually rotating the slice-thickness direction / stack normal by 90° in the UIH GUI; Gap was not the resolving factor. See [scanner validation](docs/T13_T15_SCANNER_VALIDATION.md).

Run the final sequence generator with:

```bash
bash scripts/generate_uih790_rtspiral.sh
```

The local `Pulseq` Conda environment is required. Use the official UIH `seq_to_bseq.exe` for conversion. Scanner SAR/PNS checks must not be bypassed.
