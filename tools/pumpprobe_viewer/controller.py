from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from a304data.correct import PPCorrectTool
from a304data.io import load_pp_loop
from a304data.pploopdataset import PPLoopDataset
from a304data.utils import get_closest_value


@dataclass
class LoadSummary:
    folder: str
    loop_count: int
    wavelength_min: float
    wavelength_max: float
    delay_min: float
    delay_max: float
    has_average: bool


class PumpProbeController:
    def __init__(self) -> None:
        self.dataset: PPLoopDataset | None = None
        self.view_source = "avg"

    def load_dataset(
        self,
        folder: str,
        wl_min: float,
        wl_max: float,
        read_averaged_only: bool = False,
    ) -> LoadSummary:
        ds = PPLoopDataset(
            folder=folder,
            wl_min=wl_min,
            wl_max=wl_max,
            read_averaged_only=read_averaged_only,
        )
        self.dataset = ds
        self.view_source = "avg"
        return self.get_summary()

    def load_single_file(
        self,
        path: str,
        wl_min: float,
        wl_max: float,
    ) -> LoadSummary:
        data = load_pp_loop(path, wl_min, wl_max)
        ds = SingleFilePumpProbeDataset(path, data)
        self.dataset = ds
        self.view_source = "avg"
        return self.get_summary()

    def get_summary(self) -> LoadSummary:
        ds = self._require_dataset()
        return LoadSummary(
            folder=ds.folder,
            loop_count=getattr(ds, "loop_num", 0),
            wavelength_min=float(np.nanmin(ds.wavelengths)),
            wavelength_max=float(np.nanmax(ds.wavelengths)),
            delay_min=float(np.nanmin(ds.delays)),
            delay_max=float(np.nanmax(ds.delays)),
            has_average=ds.avg_data is not None,
        )

    def clean_jump_points(
        self,
        ref_wavelength: float,
        threshold: float,
        delay_min: float | None = None,
        delay_max: float | None = None,
    ) -> None:
        ds = self._require_dataset()
        delay_range = None
        if delay_min is not None and delay_max is not None and delay_min < delay_max:
            delay_range = (delay_min, delay_max)
        ds.correct.jump_points(
            ref_wl=ref_wavelength,
            threshold=threshold,
            loop="all",
            delay_range=delay_range,
        )

    def calculate_average(self, loops_text: str = "all") -> None:
        ds = self._require_dataset()
        loops_text = loops_text.strip().lower()
        if not loops_text or loops_text == "all":
            ds.calculate_averaged_data("all")
            self.view_source = "avg"
            return

        loops = []
        for token in loops_text.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = [int(part.strip()) for part in token.split("-", 1)]
                loops.extend(range(start, end + 1))
            else:
                loops.append(int(token))
        ds.calculate_averaged_data(loops)
        self.view_source = "avg"

    def apply_chirp(self, coeffs_text: str) -> None:
        ds = self._require_dataset()
        coeffs = self.parse_coefficients(coeffs_text)
        if not coeffs:
            raise ValueError("Please enter chirp polynomial coefficients.")
        ds.correct.chirp(chirp_coeffs=coeffs)

    def subtract_background(self, before_delay: float) -> None:
        ds = self._require_dataset()
        if ds.avg_data is None:
            raise ValueError("No averaged data to subtract background from.")
        background_region = ds.avg_data.loc[ds.avg_data.index <= before_delay, :]
        if background_region.empty:
            nearest_delay = self.get_nearest_delay(before_delay)
            background = ds.avg_data.loc[nearest_delay, :]
        else:
            background = background_region.mean()
        ds.avg_data = ds.avg_data - background
        self.view_source = "avg"

    def save_current_average(self, output_name: str = "processed_averaged.dat") -> str:
        output_name = output_name.strip() or "processed_averaged.dat"
        ds = self._require_dataset()
        if ds.avg_data is None:
            raise ValueError("No averaged data to save.")
        output_path = output_name if os.path.isabs(output_name) else os.path.join(ds.folder, output_name)
        data = ds.avg_data.copy()
        data.index.name = "0"
        data.to_csv(output_path, sep="\t")
        return output_path

    def save_processed(self, output_name: str = "processed_averaged.dat") -> str:
        return self.save_current_average(output_name)

    def save_as_package_default(self) -> str:
        ds = self._require_dataset()
        ds.save_averaged_data()
        return os.path.join(ds.folder, "saved_averaged.dat")

    def get_matrix(self) -> pd.DataFrame:
        ds = self._require_dataset()
        if self.view_source.startswith("loop:"):
            loop_index = int(self.view_source.split(":", 1)[1])
            if not 1 <= loop_index <= getattr(ds, "loop_num", 0):
                raise ValueError("Selected loop is out of range.")
            return ds.data[loop_index - 1]
        if ds.avg_data is None:
            raise ValueError("No averaged data is loaded or calculated.")
        return ds.avg_data

    def set_view_source(self, source: str) -> None:
        self.view_source = source

    def list_view_sources(self) -> list[tuple[str, str]]:
        ds = self._require_dataset()
        sources = [("avg", "Averaged data")]
        loop_count = getattr(ds, "loop_num", 0)
        sources.extend((f"loop:{index}", f"Loop {index}") for index in range(1, loop_count + 1))
        return sources

    def get_view_source_label(self) -> str:
        for source, label in self.list_view_sources():
            if source == self.view_source:
                return label
        return "Averaged data"

    def get_nearest_delay(self, delay: float) -> float:
        matrix = self.get_matrix()
        return float(get_closest_value(delay, matrix.index.values))

    def get_nearest_wavelength(self, wavelength: float) -> float:
        matrix = self.get_matrix()
        return float(get_closest_value(wavelength, matrix.columns.values))

    def get_axis_labels(self) -> tuple[str, str]:
        return "Wavelength (nm)", "Delay (ps)"

    def _require_dataset(self) -> PPLoopDataset:
        if self.dataset is None:
            raise ValueError("Please load a pump-probe folder first.")
        return self.dataset

    @staticmethod
    def parse_coefficients(text: str) -> list[float]:
        cleaned = text.replace(";", ",").replace("\n", ",")
        coeffs = []
        for token in cleaned.split(","):
            token = token.strip()
            if token:
                coeffs.append(float(token))
        return coeffs


class SingleFilePumpProbeDataset:
    def __init__(self, path: str, data: pd.DataFrame) -> None:
        self.file_path = path
        self.folder = os.path.dirname(path) or os.getcwd()
        self.avg_data = data
        self.data = [data]
        self.loop_num = 1
        self.averaged_loop_num = 1
        self.averaged_loops = [1]
        self.chirp_corrected = False
        self.wavelengths = data.columns.to_numpy(dtype=float)
        self.delays = data.index.to_numpy(dtype=float)
        self.default_delays = self.delays.copy()
        self.correct = PPCorrectTool(self)
