# a304data/qbanalyzer/qbanalyzer.py

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from ..utils import get_closest_value

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a304data.pploopdataset import PPLoopDataset

class QBAnalyzer:
    """
    QBAnalyzer 的 Docstring

    提供计算量子拍信号和背景信号的若干种方法
    - 'poly': 高阶多项式拟合背景
    - 'savgol': Savitzky-Golay 滤波
    - 'manual': 手动指定参考点构建背景
    - 'fit': 扣除指数衰减背景 TODO
    - 'fft_filter': 带通 FFT 滤波 TODO
    """
    def __init__(self, dataset: "PPLoopDataset"):
        self.ds = dataset
    
    def savgol(
        self,
        window_length: int = 15,
        polyorder: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        可用的拟合背景方法之一，调用scipy.signal.savgol_filter实现。

        缺点是在零点附近会受到 artifact 明显的影响。

        Args:
            window_length (int): The length of the filter window (i.e., the number of coefficients).
            polyorder (int): The order of the polynomial used to fit the samples.
        
        Returns:
            self.ds.qb_data, self.ds.bg_data (pd.DataFrame, pd.DataFrame): 拟合的量子拍数据和背景数据。
        """
        smoothed_data = self.ds.avg_data.apply(
            lambda col: savgol_filter(col, window_length=window_length, polyorder=polyorder), axis=0
        )
        self.ds.bg_data = smoothed_data
        self.ds.qb_data = self.ds.avg_data - smoothed_data
        return self.ds.qb_data, self.ds.bg_data

    def poly(
        self,
        delay_cutoff = 0.5,
        deg = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        可用的拟合背景方法之一，指定时间零点后，用高阶多项式拟合背景，并获取量子拍信号。

        缺点是有时效果并不好，拟合出的多项式背景本身存在低频振荡。

        Args:
            delay_cutoff (float): 时间零点。
            deg (int): 多项式阶数。

        Returns:
            self.ds.qb_data, self.ds.bg_data (pd.DataFrame, pd.DataFrame): 拟合的量子拍数据和背景数据。
        """
        bg_data = pd.DataFrame(index=self.ds.delays, columns=self.ds.wavelengths, dtype=float)
        bg_data.index.name = '0'

        for wl in self.ds.wavelengths:
            data_segment = self.ds.avg_data.loc[delay_cutoff: , wl].dropna()
            if len(data_segment) < 3:
                bg_data[wl] = np.nan
                continue
            fit_coeffs = np.polyfit(data_segment.index.values, data_segment.values, deg=deg)
            bg_data[wl] = np.poly1d(fit_coeffs)(self.ds.delays)
        
        self.ds.bg_data = bg_data
        self.ds.qb_data = self.ds.avg_data - bg_data
        return self.ds.qb_data, self.ds.bg_data

    def manual(
        self,
        ref_wl: float,
        ref_delay: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        不推荐的拟合背景方法。或者说根本不算是拟合。

        Args:
            ref_wl (float): 指定参考点的波长。
            ref_delay (float): 指定参考点的 delay。
        
        Returns:
            self.ds.qb_data, self.ds.bg_data (pd.DataFrame, pd.DataFrame): 拟合的量子拍数据和背景数据。
        """
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
        self.ds.bg_data = bg_data
        self.ds.qb_data = self.ds.avg_data - bg_data
        return self.ds.qb_data, self.ds.bg_data

    def fit(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def fft_filter(
        self,
        cutoff_low: float = 0.1,
        cutoff_high: float = 2.0,
        width: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
