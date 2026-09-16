# UIH uMR 790 real-time spiral bSSFP

This project adapts USC RTSpiral to a UIH uMR 790 fixed-slice, 2D multi-slice real-time trueFISP/bSSFP acquisition.  It retains the USC VDS spiral, M1-nulled `gropt` rewinder, and golden-angle rotation while adding UIH timing, no-trigger acquisition, SLC/REP/LIN labels, UIH Definitions, and output provenance.

Run the final sequence generator with:

```bash
bash scripts/generate_uih790_rtspiral.sh
```

The local `Pulseq` Conda environment is required.  The generated `.seq` must be checked in UIH AIDE / Pulseq Virtual Scan before scanner use; local generation does not validate SAR, PNS, B1/100-degree FA, or raw-data label mapping.
