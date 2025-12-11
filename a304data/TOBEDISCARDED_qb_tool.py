################## 准备废弃 ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Callable
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from .utils import get_closest_value

class QBTool:
    """
    Quantum Beats (QB) calculation tool.

    支持多种方法计算量子振荡信号：
    - 'fit': 扣除指数衰减背景
    - 'fft_filter': 带通 FFT 滤波
    - 'savgol': Savitzky-Golay 滤波
    - 'manual': 手动指定参考点构建背景
    """

    def __init__(self, dataset):
        """
        Args:
            dataset: PPLoopDataset 实例，包含 avg_data, delays, wavelengths
        """
        self.dataset = dataset

    def calculate(
        self,
        method: Literal['fit', 'fft_filter', 'savgol', 'manual'] = 'fit',
        **kwargs
    ) -> pd.DataFrame:
        """
        计算量子振荡数据。

        Args:
            method: 计算方法
            kwargs: 各方法可选参数（IDE 会显示提示）

        Returns:
            pd.DataFrame: 计算得到的量子振荡数据
        """
        method_map = {
            'fit': self._fit_method,
            'fft_filter': self._fft_filter_method,
            'savgol': self._savgol_method,
            'manual': self._manual_method,
        }

        if method not in method_map:
            raise ValueError(f"Unsupported method: {method}. Choose from {list(method_map.keys())}")

        self.dataset.qb_data = method_map[method](**kwargs)
        return self.dataset.qb_data

    # --------------- 内部方法 ---------------
    def _fit_method(self) -> pd.DataFrame:
        """
        扣除拟合的指数衰减背景
        """
        if not hasattr(self.dataset, 'decay'):
            raise RuntimeError("Please run dataset.calculate_decay() first.")

        fitted_data = pd.DataFrame(index=self.dataset.delays, columns=self.dataset.wavelengths)
        fitted_data.index.name = '0'
        for wl in self.dataset.wavelengths:
            bg_coeff = self.dataset.decay[wl][0]
            if bg_coeff is None:
                fitted_data[wl] = np.nan
            else:
                fitted_data[wl] = np.poly1d(bg_coeff)(self.dataset.delays)
        return self.dataset.avg_data - fitted_data

    def _fft_filter_method(
        self,
        cutoff_low: float = 0.1,
        cutoff_high: float = 2.0,
        width: float = 0.15
    ) -> pd.DataFrame:
        """
        FFT 带通滤波去掉低频背景和高频噪声
        """
        dt = np.mean(np.diff(self.dataset.delays))
        N = len(self.dataset.delays)
        freq = fftfreq(N, dt)

        def soft_edge(x, f0, w):
            return 0.5 * (1 + np.tanh((np.abs(x) - f0) / w))

        highpass = soft_edge(freq, cutoff_low, width)
        lowpass = 1 - soft_edge(freq, cutoff_high, width)
        bandpass = highpass * lowpass

        qb_data = pd.DataFrame(index=self.dataset.delays, columns=self.dataset.wavelengths, dtype=float)
        qb_data.index.name = '0'

        for wl in self.dataset.wavelengths:
            signal = self.dataset.avg_data[wl].values.copy()
            mask = self.dataset.delays < 0.5
            if np.any(mask):
                valid = (~mask) & (self.dataset.delays < 1.0)
                if valid.sum() > 1:
                    p = np.polyfit(self.dataset.delays[valid], signal[valid], 1)
                    signal[mask] = np.polyval(p, self.dataset.delays[mask])
                else:
                    signal[mask] = signal[~mask][0]
            filtered_signal = np.real(ifft(fft(signal) * bandpass))
            qb_data[wl] = filtered_signal

        return qb_data

    def _savgol_method(self, window_length: int = 11, polyorder: int = 3) -> pd.DataFrame:
        """
        使用 Savitzky-Golay 滤波去背景
        """
        smoothed_data = self.dataset.avg_data.apply(
            lambda col: savgol_filter(col, window_length=window_length, polyorder=polyorder), axis=0
        )
        return self.dataset.avg_data - smoothed_data

    def _manual_method(self, ref_wl: float = 485, ref_delay: float = 1.0) -> pd.DataFrame:
        """
        手动背景扣除方法，使用参考点
        """
        m_wl = get_closest_value(ref_wl, self.dataset.wavelengths)
        m_delay = get_closest_value(ref_delay, self.dataset.delays)
        bg_data = pd.DataFrame(index=self.dataset.delays, columns=self.dataset.wavelengths, dtype=float)
        bg_data.index.name = '0'

        for wl in self.dataset.wavelengths:
            for delay in self.dataset.delays:
                bg_data.at[delay, wl] = (
                    self.dataset.avg_data.at[m_delay, wl] * self.dataset.avg_data.at[delay, m_wl]
                    / self.dataset.avg_data.at[m_delay, m_wl]
                )
        return self.dataset.avg_data - bg_data

