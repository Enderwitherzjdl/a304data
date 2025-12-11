# a304data/qbanalyzer/qbanalyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Callable
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from ..utils import get_closest_value

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a304data.pploopdataset import PPLoopDataset

class QBAnalyzer:
    """
    QBAnalyzer 的 Docstring

    提供计算量子拍信号的若干种方法
    - 'savgol': Savitzky-Golay 滤波
    - 'manual': 手动指定参考点构建背景
    - 'fit': 扣除指数衰减背景 TODO
    - 'fft_filter': 带通 FFT 滤波 TODO
    """
    def __init__(self, dataset: "PPLoopDataset"):
        self.ds = dataset
    
    def savgol(
        self,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> pd.DataFrame:
        smoothed_data = self.ds.avg_data.apply(
            lambda col: savgol_filter(col, window_length=window_length, polyorder=polyorder), axis=0
        )
        self.ds.qb_data = self.ds.avg_data - smoothed_data
        return self.ds.qb_data

    def manual(
        self,
        ref_wl: float,
        ref_delay: float,
    ) -> pd.DataFrame:
        m_wl = get_closest_value(ref_wl, self.ds.wavelengths)
        m_delay = get_closest_value(ref_delay, self.ds.delays)
        bg_data = pd.DataFrame(index=self.ds.delays, columns=self.ds.wavelengths, dtype=float)
        bg_data.index.name = '0'

        for wl in self.ds.wavelengths:
            for delay in self.ds.delays:
                bg_data.at[delay, wl] = (
                    self.ds.avg_data.at[m_delay, wl] * self.ds.avg_data.at[delay, m_wl]
                    / self.ds.avg_data.at[m_delay, m_wl]
                )
        self.ds.qb_data = self.ds.avg_data - bg_data
        return self.ds.qb_data

    def fit(
        self,
    ) -> pd.DataFrame:
        pass

    def fft_filter(
        self,
        cutoff_low: float = 0.1,
        cutoff_high: float = 2.0,
        width: float = 0.15
    ) -> pd.DataFrame:
        pass