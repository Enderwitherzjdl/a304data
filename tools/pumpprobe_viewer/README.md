# Pump-Probe Loop Processor

Desktop UI for the common pump-probe loop workflow:

1. Choose a data folder.
2. Loop files or an existing averaged file load automatically.
3. Clean jump points from all loops.
4. Average selected loops.
5. Apply chirp correction from polynomial coefficients.
6. Optionally subtract a pre-zero background.
7. Save the current averaged matrix in the same tab-separated format.

## Run

From the repository root:

```powershell
python tools\pumpprobe_viewer\main.py
```

Or double-click the shortcut:

```text
tools\pumpprobe_viewer\Pump-Probe Loop Processor.lnk
```

## Notes

- Chirp coefficients are entered highest-order first, matching `numpy.poly1d`.
  The polynomial should return the absolute chirp delay for each wavelength;
  no reference wavelength is subtracted.
- `loops` accepts `all`, comma-separated loop numbers such as `1,2,5`, or ranges
  such as `1-5,8`.
- The `view data` selector keeps `Averaged data` at the top, followed by each
  loaded loop.
- The delay axis can use signed-log display: zero stays fixed, while positive
  and negative delays are compressed separately.
- In the 2D map, Ctrl+click adds a probe slice and Shift+click adds a delay
  slice.
- `Save current average` writes the filename shown in the save box, defaulting to
  `processed_averaged.dat`.
