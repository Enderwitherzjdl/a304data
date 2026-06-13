import os
import re
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_2dvis_data
from .utils import get_closest_value


class TwoDVisDataset:
    """
    2D-VIS data set.

    The expected folder layout is the format produced by the current 2D-VIS
    acquisition program:

    - Raw_Data: files named like ``260429_124622_-0.05ps_raw.dat``
    - Auto_FFT: files named like ``260429_124622_-0.05ps_fft.dat``

    Each file is a matrix. The first row stores probe wavelengths, the first
    column stores the raw scan axis or FFT frequency axis, and the waiting time
    is parsed from the file name.
    """

    def __init__(
        self,
        folder,
        wl_min=None,
        wl_max=None,
        read_raw=True,
        read_fft=True,
    ):
        self.folder = folder
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.pump_wl = self._extract_pump_wl(os.path.basename(os.path.abspath(folder)))

        self.raw_data = {}
        self.fft_data = {}
        self.raw_avg_data = {}
        self.fft_avg_data = {}

        if read_raw:
            self._load_original_data("raw")
        if read_fft:
            self._load_original_data("fft")
        if not self.raw_data and not self.fft_data:
            raise ValueError(f"Folder {self.folder} does not contain 2D-VIS data.")

        self.waiting_times = self._collect_waiting_times()
        first = self._first_loaded_frame()
        self.wavelengths = first.columns.values
        self.default_wavelengths = self.wavelengths.copy()

        if self.raw_data:
            self.raw_axis = self._first_frame(self.raw_data).index.values
        else:
            self.raw_axis = None
        if self.fft_data:
            self.fft_axis = self._first_frame(self.fft_data).index.values
        else:
            self.fft_axis = None

    def _extract_pump_wl(self, path: str) -> int | None:
        s = path.lower()
        m = re.search(r"pump\s*([0-9]{2,5})", s)
        if m:
            return int(m.group(1))
        m = re.search(r"([0-9]{2,5})\s*pump", s)
        if m:
            return int(m.group(1))
        m = re.search(r"([0-9]{2,5})\s*nm", s)
        if m:
            return int(m.group(1))
        return None

    def _extract_waiting_time(self, filename: str) -> float:
        m = re.search(r"_(-?\d+(?:\.\d+)?)ps_", filename)
        if m is None:
            raise ValueError(f"Cannot parse waiting time from {filename}.")
        return float(m.group(1))

    def _data_dir(self, kind: Literal["raw", "fft"]) -> str:
        if kind == "raw":
            return os.path.join(self.folder, "Raw_Data")
        if kind == "fft":
            return os.path.join(self.folder, "Auto_FFT")
        raise ValueError("kind should be 'raw' or 'fft'.")

    def _load_original_data(self, kind: Literal["raw", "fft"]):
        data_dir = self._data_dir(kind)
        if not os.path.isdir(data_dir):
            return False

        suffix = f"_{kind}.dat"
        target = self.raw_data if kind == "raw" else self.fft_data
        files = sorted(f for f in os.listdir(data_dir) if f.endswith(suffix))
        for file in files:
            waiting_time = self._extract_waiting_time(file)
            full_path = os.path.join(data_dir, file)
            target.setdefault(waiting_time, []).append(
                load_2dvis_data(full_path, self.wl_min, self.wl_max)
            )

        count = sum(len(v) for v in target.values())
        if count:
            print(f"Load {kind} 2D-VIS files: {count} scans, {len(target)} waiting times.")
        return bool(count)

    def _collect_waiting_times(self):
        times = set(self.raw_data.keys()) | set(self.fft_data.keys())
        return np.array(sorted(times), dtype=float)

    def _first_frame(self, data_dict):
        for waiting_time in sorted(data_dict):
            if data_dict[waiting_time]:
                return data_dict[waiting_time][0]
        raise ValueError("No 2D-VIS frame loaded.")

    def _first_loaded_frame(self):
        if self.raw_data:
            return self._first_frame(self.raw_data)
        return self._first_frame(self.fft_data)

    def _select_source(self, kind: Literal["raw", "fft"], averaged=False):
        if kind == "raw":
            return self.raw_avg_data if averaged else self.raw_data
        if kind == "fft":
            return self.fft_avg_data if averaged else self.fft_data
        raise ValueError("kind should be 'raw' or 'fft'.")

    def _normalize_waiting_times(self, waiting_times):
        if isinstance(waiting_times, str) and waiting_times == "all":
            return list(self.waiting_times)
        if isinstance(waiting_times, (int, float)):
            return [float(waiting_times)]
        return [float(t) for t in waiting_times]

    def _nearest_waiting_time(self, waiting_time):
        return get_closest_value(waiting_time, self.waiting_times)

    # --------------- Calculate Data ---------------
    def calculate_averaged_data(self, kind: Literal["raw", "fft", "all"] = "all", waiting_times="all"):
        """
        Average repeated scans grouped by waiting time.

        Args:
            kind: ``"raw"``, ``"fft"``, or ``"all"``.
            waiting_times: ``"all"``, one waiting time, or an iterable of
                waiting times in ps.
        """
        kinds = ("raw", "fft") if kind == "all" else (kind,)
        selected_times = self._normalize_waiting_times(waiting_times)

        for current_kind in kinds:
            source = self._select_source(current_kind, averaged=False)
            target = self._select_source(current_kind, averaged=True)
            for waiting_time in selected_times:
                nearest_time = self._nearest_waiting_time(waiting_time)
                if nearest_time not in source:
                    continue
                target[nearest_time] = pd.concat(source[nearest_time]).groupby(level=0).mean()
            print(f"Calculate averaged {current_kind} data: {len(target)} waiting times.")

    # --------------- Access Data ---------------
    def get_data(
        self,
        waiting_time,
        kind: Literal["raw", "fft"] = "fft",
        averaged=True,
        scan_index=1,
    ):
        """
        Return a 2D matrix by waiting time.

        Args:
            waiting_time: Waiting time in ps. The closest loaded value is used.
            kind: ``"raw"`` or ``"fft"``.
            averaged: Read from averaged data when True.
            scan_index: 1-based scan index used when ``averaged`` is False.
        """
        source = self._select_source(kind, averaged=averaged)
        nearest_time = self._nearest_waiting_time(waiting_time)
        if nearest_time not in source:
            raise ValueError(f"No {kind} data for waiting time {nearest_time} ps.")
        data = source[nearest_time]
        if averaged:
            return data
        if not 1 <= scan_index <= len(data):
            raise ValueError("scan_index out of range.")
        return data[scan_index - 1]

    def get_spectrum(self, waiting_time, axis_value, kind: Literal["raw", "fft"] = "fft", averaged=True):
        """Return one wavelength spectrum at the closest raw/FFT axis value."""
        data = self.get_data(waiting_time, kind=kind, averaged=averaged)
        axis = get_closest_value(axis_value, data.index.values)
        return data.loc[axis, :]

    def get_trace(self, waiting_time, wavelength, kind: Literal["raw", "fft"] = "fft", averaged=True):
        """Return one raw/FFT-axis trace at the closest probe wavelength."""
        data = self.get_data(waiting_time, kind=kind, averaged=averaged)
        wl = get_closest_value(wavelength, data.columns.values)
        return data.loc[:, wl]

    # --------------- Save Data ---------------
    def save_averaged_data(self, kind: Literal["raw", "fft", "all"] = "all"):
        """
        Save averaged matrices into ``Saved_Averaged`` under the data folder.
        """
        kinds = ("raw", "fft") if kind == "all" else (kind,)
        for current_kind in kinds:
            source = self._select_source(current_kind, averaged=True)
            if not source:
                raise ValueError(f"No averaged {current_kind} data calculated.")
            save_dir = os.path.join(self.folder, "Saved_Averaged", current_kind)
            os.makedirs(save_dir, exist_ok=True)
            for waiting_time, data in source.items():
                save_path = os.path.join(save_dir, f"{waiting_time:g}ps_{current_kind}_averaged.dat")
                data.to_csv(save_path, sep="\t")
            print(f"Saved averaged {current_kind} data to {save_dir}")

    # --------------- Plot Data ---------------
    def plot_map(
        self,
        waiting_time,
        kind: Literal["raw", "fft"] = "fft",
        averaged=True,
        cmap="RdBu_r",
        vlim: Literal["maxmin"] | float | None = "maxmin",
        xlim=None,
        ylim=None,
        savefig=False,
    ):
        """Plot one 2D-VIS matrix as a wavelength-axis map."""
        data = self.get_data(waiting_time, kind=kind, averaged=averaged)
        x = data.columns.values
        y = data.index.values
        z = data.values

        if vlim == "maxmin":
            vmax = np.nanmax(np.abs(z))
            vmin = -vmax
        elif isinstance(vlim, (int, float)):
            vmax = abs(vlim)
            vmin = -vmax
        else:
            vmin = vmax = None

        plt.pcolormesh(x, y, z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlabel("Probe wavelength (nm)", fontsize=14)
        plt.ylabel("Raw axis" if kind == "raw" else "FFT axis", fontsize=14)
        plt.title(f"2D-VIS {kind.upper()} {self._nearest_waiting_time(waiting_time):g} ps", fontsize=14)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if savefig:
            name = f"2dvis_{kind}_{self._nearest_waiting_time(waiting_time):g}ps.png"
            plt.savefig(os.path.join(self.folder, name))
        plt.show()
