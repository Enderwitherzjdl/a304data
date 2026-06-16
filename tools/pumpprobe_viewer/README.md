# Pump-Probe Loop Processor

Desktop UI for the common pump-probe loop workflow:

1. Choose a data folder or a single data file.
2. Loop files, an existing averaged file, or the chosen single file load automatically.
3. Inspect single loops and clear selected jump points by eye.
4. Average selected loops after the inspected loops look clean.
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
- In a single loop view, use `Clear jump point`, click a jump point, then press
  `Confirm clear` to replace that delay row with the average of its neighbors.
- The `FIGURE` controls limit the displayed delay and wavelength ranges without
  changing the loaded data.
- `Choose Data` opens one `.dat` or `.txt` matrix directly and treats it as the
  current averaged dataset.
- `Save current average` opens a save dialog and appends `.dat` when needed.
