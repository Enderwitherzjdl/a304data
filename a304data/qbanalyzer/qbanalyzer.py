# a304data/qbanalyzer/qbanalyzer.py

import numpy as np
import pandas as pd
from ..utils import get_closest_value

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a304data.pploopdataset import PPLoopDataset

class QBAnalyzer:
    """
    QBAnalyzer 的 Docstring

    提供计算量子拍信号和背景信号的若干种方法
    - 'savgol': Savitzky-Golay 滤波
    - 'poly': 高阶多项式拟合背景
    - 'exp': 用1-2个指数拟合背景

    提供基于量子拍信号，做FFT得到频谱的方法
    - 'fft': 朴素的FFT TODO
    """
    def __init__(self, dataset: "PPLoopDataset"):
        self.ds = dataset

    # ================= QB 区 =================    
    # ----------------- 可用的 -----------------
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
        from scipy.signal import savgol_filter
        smoothed_data = self.ds.avg_data.apply(
            lambda col: savgol_filter(col, window_length=window_length, polyorder=polyorder), axis=0
        )
        self.ds.bg_data = smoothed_data
        self.ds.qb_data = self.ds.avg_data - smoothed_data
        self.ds.qb_method = 'Savitzky-Golay filter'
        return self.ds.qb_data, self.ds.bg_data

    def poly(
        self,
        delay_cutoff = 0,
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
        bg_data = pd.DataFrame(index=self.ds.default_delays, columns=self.ds.wavelengths, dtype=float)
        bg_data.index.name = '0'

        for wl in self.ds.wavelengths:
            data_segment = self.ds.avg_data.loc[delay_cutoff + self.ds.delay_zero: , wl].dropna()
            if len(data_segment) < 3:
                bg_data[wl] = np.nan
                continue
            fit_coeffs = np.polyfit(data_segment.index.values, data_segment.values, deg=deg)
            bg_data[wl] = np.poly1d(fit_coeffs)(self.ds.default_delays)
        
        self.ds.bg_data = bg_data
        self.ds.qb_data = self.ds.avg_data - bg_data
        self.ds.qb_method = 'Polynomial'
        return self.ds.qb_data, self.ds.bg_data

    def exp(
        self,
        delay_cutoff: float = 0,
        n_exp: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定时间零点后，用 1–3 个指数衰减函数拟合背景，并获取量子拍信号。3 个及以上的指数拟合物理上病态严重，最好不要使用。

        优化了拟合的参数初猜，在数据生成初猜参数、邻近波长拟合参数中，取结果更好的那个。

        背景模型：
            n_exp = 1: A1 * exp(-t/tau1) + C
            n_exp = 2: A1 * exp(-t/tau1) + A2 * exp(-t/tau2) + C
            n_exp = 3: A1 * exp(-t/tau1) + A2 * exp(-t/tau2) + A3 * exp(-t/tau3) + C

        Args:
            delay_cutoff (float): 时间零点。
            n_exp (int): 指数个数（1–3）。

        Returns:
            self.ds.qb_data, self.ds.bg_data (pd.DataFrame, pd.DataFrame)
        """
        from scipy.optimize import curve_fit

        if n_exp not in (1, 2, 3):
            raise ValueError("n_exp must be 1, 2, or 3.")
        if n_exp >= 3:
            print(f'Warning: n_exp > 3 may cause physical errors!')

        bg_data = pd.DataFrame(index=self.ds.default_delays, columns=self.ds.wavelengths, dtype=float)
        bg_data.index.name = '0'

        def _exp_func_factory(n):
            def func(t, *params):
                y = np.zeros_like(t, dtype=float)
                for i in range(n):
                    A = params[2 * i]
                    tau = params[2 * i + 1]
                    y += A * np.exp(-t / tau)
                C = params[-1]
                return y + C
            return func
        exp_func = _exp_func_factory(n_exp)

        def _initial_guess(y):
            guess = []
            amp = y[0] / n_exp
            for i in range(n_exp):
                guess += [amp, 10**i]
            guess += [0.0]  # offset C
            return guess

        def _fit_with_p0(p0):
            params, _ = curve_fit(
                exp_func,
                t,
                y,
                p0=p0,
                maxfev=10000,
            )
            residual = y - exp_func(t, *params)
            sse = np.nansum(residual ** 2)
            return params, sse

        # === 主循环 ===
        last_p0 = None
        for wl in self.ds.wavelengths:
            data_segment = self.ds.avg_data.loc[delay_cutoff + self.ds.delay_zero:, wl].dropna()
            if len(data_segment) < 2 * n_exp + 1:
                bg_data[wl] = np.nan
                continue
            t = data_segment.index.values
            y = data_segment.values

            best_params = None; best_sse = np.inf
            p0_candidates = [
                last_p0,               # 上一个波长的拟合参数
                _initial_guess(y),     # 数据生成初猜参数
            ]
            for p0 in p0_candidates:
                if p0 is None:
                    continue
                try:
                    params, sse = _fit_with_p0(p0)
                    if sse < best_sse:
                        best_sse = sse
                        best_params = params
                except RuntimeError:
                    continue
            if best_params is not None:
                last_p0 = best_params
                bg_data[wl] = exp_func(self.ds.default_delays, *best_params)
            else:
                bg_data[wl] = np.nan

        # === 保存 & 返回 ===
        self.ds.bg_data = bg_data
        self.ds.qb_data = self.ds.avg_data - bg_data
        self.ds.qb_method = 'Exponential'
        return self.ds.qb_data, self.ds.bg_data

    # ----------------- 不推荐 -----------------
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
        m_delay = get_closest_value(ref_delay + self.ds.delay_zero, self.ds.default_delays)
        bg_data = pd.DataFrame(index=self.ds.default_delays, columns=self.ds.wavelengths, dtype=float)
        bg_data.index.name = '0'

        for wl in self.ds.wavelengths:
            for delay in self.ds.default_delays:
                bg_data.at[delay, wl] = (
                    self.ds.avg_data.at[m_delay, wl] * self.ds.avg_data.at[delay, m_wl]
                    / self.ds.avg_data.at[m_delay, m_wl]
                )
        self.ds.bg_data = bg_data
        self.ds.qb_data = self.ds.avg_data - bg_data
        return self.ds.qb_data, self.ds.bg_data

    def svd(
        self,
        delay_cutoff: float = 0.5,
        n_comp: int = 2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        svd 的 Docstring
        
        gpt写的SVD，感觉没法用。
        """
        if n_comp < 1:
            raise ValueError("n_comp must be >= 1.")

        # === 1. 截取数据 ===
        data_seg = self.ds.avg_data.loc[delay_cutoff + self.ds.delay_zero:, :]

        # 删除全 NaN 列
        data_seg = data_seg.dropna(axis=1, how="all")

        if data_seg.empty:
            raise ValueError("No valid data after delay_cutoff.")

        D = data_seg.values.astype(float)

        # === 2. 处理 NaN / Inf ===
        mask = ~np.isfinite(D)
        if mask.any():
            # 用列均值填充
            col_mean = np.nanmean(D, axis=0)
            D[mask] = np.take(col_mean, np.where(mask)[1])

        # === 3. 删除几乎无变化的列（关键）===
        col_std = np.std(D, axis=0)
        valid_cols = col_std > 1e-12
        D = D[:, valid_cols]
        cols_kept = data_seg.columns[valid_cols]

        if D.shape[1] < n_comp:
            raise ValueError("Not enough valid columns for SVD.")

        # === 4. 去均值 + 归一化 ===
        mean = D.mean(axis=0, keepdims=True)
        std = D.std(axis=0, keepdims=True)
        std[std == 0] = 1.0

        D0 = (D - mean) / std

        # === 5. SVD（加保护）===
        try:
            U, S, Vt = np.linalg.svd(D0, full_matrices=False)
        except np.linalg.LinAlgError:
            # fallback：降低精度再试
            U, S, Vt = np.linalg.svd(D0.astype(np.float32), full_matrices=False)

        # === 6. 重构背景 ===
        D_bg_seg = (U[:, :n_comp] * S[:n_comp]) @ Vt[:n_comp, :]
        D_bg_seg = D_bg_seg * std + mean

        # === 7. 放回完整矩阵 ===
        bg_data = pd.DataFrame(
            index=self.ds.default_delays,
            columns=self.ds.wavelengths,
            dtype=float,
        )
        bg_data.index.name = '0'
        bg_data.loc[data_seg.index, cols_kept] = D_bg_seg

        # 时间零点前外推
        bg_data.loc[:delay_cutoff, cols_kept] = bg_data.loc[delay_cutoff, cols_kept].values

        # === 8. QB 信号 ===
        self.ds.bg_data = bg_data
        self.ds.qb_data = self.ds.avg_data - bg_data
        self.ds.qb_method = 'SVD'
        return self.ds.qb_data, self.ds.bg_data

    # ----------------- 施工中 -----------------





    # ================= FFT 区 =================

    def fft(self, delay_range=(0.15,15), N=1000):
        """
        对量子拍做 FFT，获取振动的频率分布。

        Args:
            delay_range (tuple[float,float]): 做 FFT 的时间区间。
            N (int): 采样点数，别太小就行。
        """
        if self.ds.qb_data is None:
            raise ValueError(f'Calculate qb_data first!')
        from scipy.interpolate import interp1d
        from scipy.fft import fft, fftfreq
        
        t = np.linspace(delay_range[0]+self.ds.delay_zero, delay_range[1]+self.ds.delay_zero, N)
        dt = t[1]-t[0]
        freqs = fftfreq(N, dt) * 33.35640952 # 单位是THz，默认转换为cm-1
        self.ds.freqs = freqs[:(N+1)>>1] # 只保留非负频率
        
        fft_data = pd.DataFrame(index=self.ds.freqs, columns=self.ds.wavelengths, dtype=float)
        for wl in self.ds.wavelengths:
            # fill_value=0，可能可以用来延长时间范围，提高分辨率
            interp_func = interp1d(self.ds.default_delays, self.ds.qb_data[wl], kind='linear', bounds_error=False, fill_value=0) 
            interp_values = interp_func(t)
            fft_values = fft(interp_values) # norm='forward'or'ortho'?
            fft_data[wl] = fft_values[:(N+1)>>1]
        self.ds.fft_data = fft_data