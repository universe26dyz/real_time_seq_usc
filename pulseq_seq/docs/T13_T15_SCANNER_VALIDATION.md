# T13--T15 scanner validation

T13, T14, and T15 were each run successfully on 5-slice and 60-slice stacks on the scanner.

- Scanner acquisition is 350 continuous arms per fixed slice.
- Temporal frames are reconstruction-only: 7 consecutive arms are grouped into one frame, yielding 50 derived frames per slice. No temporal-frame scanner label is written.
- The overlapping stack uses physical-support FOV-z: `(N - 1) × slice shift + slice thickness`.
- The former 60-slice UIH warning, `The stack is out of the linear gradient area.`, disappeared after manually rotating the slice-thickness direction / stack normal by 90° in the UIH GUI.
- Changing Gap was not the resolving factor. An earlier hypothesis that Gap could affect the warning is retained only as history.

This is a scanner observation, not a confirmed mechanism for why the 90° rotation changes UIH warning behaviour.
