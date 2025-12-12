import os,re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing import Literal
from .io import load_pp_loop
from .utils import get_closest_value

from .info_manager import PPInfoManager
from .plot import PPPlotTool
from .correct import PPCorrectTool
from .qbanalyzer import QBAnalyzer

class PPLoopDataset:
    """
    Pump-Probe 数据集类。（TODO 注释要重写）
    
    创建时指定目录 folder 和分析的波长范围 wl_min, wl_max。
    - 可见区：建议 450~750 nm。
    - 近红外区：建议 900~1200 nm。

    支持读取每一圈的原始数据。
    - 格式一般为 {yymmdd_hhmmss_loop#.dat}
    - 可以对原始数据进行清除坏点、计算平均值等操作。

    支持读取 Labview 程序保存的，或本程序包保存的平均值文件。
    - Labview 程序保存的，文件名为 {yymmdd_hhmmss_averaged.dat}
    - 本程序包保存的，文件名为 {saved_averaged.dat}
    - 【开发中】支持对平均值数据进行 chirp 校正。

    支持数据的绘图和保存。
    - 绘图时，可以指定绘图对象，可以是'avg'、loop编号以及据此构成的数组。
    - 可以按照波长、按照延时、按照圈数绘图。【二维热图开发中】
    - 关键字 savefig=True 可以保存图片。
    """
    def __init__(self, folder, wl_min, wl_max, read_averaged_only=False):
        self.folder = folder
        self.pump_wl = self._extract_pump_wl(os.path.basename(os.path.abspath(folder)))
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.type = 'VIS'
        self.info_mgr = PPInfoManager(folder)
        self.qb = QBAnalyzer(self)             # 挂载 量子拍qb 工具
        self.plot = PPPlotTool(self)           # 挂载 绘图plot 工具
        self.correct = PPCorrectTool(self)     # 挂载 校正correct 工具
        # 加载数据
        self.data = []
        if read_averaged_only == False:
            self._load_original_data()
        self._load_averaged_data()
        # 获取数据集基本信息
        if self.data:
            self.delays = self.data[0].index.values
            self.wavelengths = self.data[0].columns.values
        else:
            self.delays = self.avg_data.index.values
            self.wavelengths = self.avg_data.columns.values

    def _extract_pump_wl(self, path: str) -> int | None:
        s = path.lower()
        m = re.search(r"pump\s*([0-9]{2,5})", s)
        if m:
            return int(m.group(1))
        m = re.search(r"([0-9]{2,5})\s*nm", s)
        if m:
            return int(m.group(1))
        return None

    # --------------- Load Data ---------------
    def _check_saved_averaged_file(self):
        """
        检查并返回平均值文件。

        优先读取本程序保存的 saved_averaged.dat；
        若不存在，则检查 Labview 程序保存的 averaged.dat；
        若存在多个 averaged 文件则报错。

        Returns:
            str | None: 找到的平均值文件名；若未找到返回 None。

        Raises:
            ValueError: 当检测到多个 averaged 文件时。
        """
        count = 0
        file_list = [f for f in os.listdir(self.folder) if 'averaged' in f]
        for file in file_list:
            if 'saved_averaged' in file:
                return file
            count += 1
        if count == 0:
            return None
        if count > 1:
            raise ValueError(f"Too many original averaged files!")
        return file_list[0]
    
    def _load_averaged_data(self):
        """
        读取平均值数据及其信息。

        优先通过 _check_saved_averaged_file() 获取文件名。
        若存在 saved_info.dat 文件，则加载其中的平均圈数信息.

        Returns:
            bool: 若未找到平均值文件返回 False，否则返回 None。
        """
        avg_file = self._check_saved_averaged_file()
        if avg_file is None:
            print("Averaged file not found.")
            return False
        avg_path = os.path.join(self.folder, avg_file)
        self.avg_data = load_pp_loop(avg_path, self.wl_min, self.wl_max)

        self.averaged_loop_num = self.info_mgr.get("data", "averaged_loop_num")
        self.averaged_loops = self.info_mgr.get("data", "averaged_loops",)
        self.chirp_corrected = self.info_mgr.get("chirp", "corrected", False)

        print(f'Load averaged file from {avg_path}')
        print(f'Averaged data chirp corrected: {self.chirp_corrected}')

    def _load_original_data(self):
        """
        加载目录中的所有原始 loop 数据文件。

        文件名中需包含 '_loop'（可见近红外）或 'pp'（红外）。
        使用 load_pp_loop() 读取各文件并保存至 self.data。
        若未检测到任何 loop 数据则报错。

        Raises:
            ValueError: 当目录中未找到任何 loop 数据文件时。
        """
        file_list = [f for f in os.listdir(self.folder) if '_loop' in f or 'pp' in f]
        if 'pp' in file_list[0]: # 通过文件命名判定数据来源
            self.type = 'IR'
        else:
            self.type = 'VIS'
        for file in file_list:
            full_path = os.path.join(self.folder, file)
            self.data.append(load_pp_loop(full_path, self.wl_min, self.wl_max))
        if not self.data:
            raise ValueError(f"Folder {self.folder} does not contain any loop data.")
        self.loop_num = len(self.data)
        print(f'Load original files: {self.loop_num} loops.')
    
    def _load_chirp_data(self,chirp_dir):
        """
        加载 chirp 数据。只允许使用 chirp 数据计算 chirp。

        Args:
            chirp_dir (str): chirp 文件目录，包含有且仅有一个 'averaged' 文件。
        """
        files = [f for f in os.listdir(chirp_dir) if 'averaged' in f]
        if len(files) != 1:
            raise ValueError(f'Not exactly one chirp file.')
        self.chirp_data = load_pp_loop(os.path.join(chirp_dir,files[0]), self.wl_min, self.wl_max)

    # --------------- Calculate Data ---------------

    def calculate_averaged_data(self, index='all'):
        """
        计算平均数据。
        
        如绘图发现跳点，请用 clean_jump_points() 清除。

        Args:
            index (str | int | list | tuple | np.ndarray):
                - 'all': 所有圈求平均；
                - int: 取前 index 个圈；
                - list/tuple/array: 按指定圈编号求平均。

        Raises:
            ValueError: index 超出范围或非法时。
        """
        if index == 'all':
            selected_data = self.data
        elif isinstance(index, int):
            if 0 < index <= self.loop_num:
                selected_data = self.data[:index]
            else:
                raise ValueError(f"Loop index out of range.")
        else:
            idx_arr = np.array(index)
            if idx_arr.max() > self.loop_num or idx_arr.min() <= 0:
                raise ValueError(f"Loop index out of range.")
            idx_arr -= 1
            selected_data = [self.data[i] for i in idx_arr]
        self.avg_data = pd.concat(selected_data).groupby(level=0).mean()
        if index == 'all':
            self.averaged_loop_num = self.loop_num
            self.averaged_loops = list(range(1, self.loop_num+1))
        elif isinstance(index, int):
            self.averaged_loop_num = index
            self.averaged_loops = list(range(1, index+1))
        else:
            self.averaged_loop_num = len(index)
            self.averaged_loops = list(index)
        self.chirp_corrected = False # 新计算的平均值是没有被校正过的
        print(f'Calculate averaged data from {self.averaged_loop_num} loops.')

    def calculate_chirp(
        self, 
        check_function=lambda _,__: True,
        deg=3,
        plot=False,
    ):
        """
        计算数据中的 chirp。

        Args:
            check_function (lambda): 自定义chirp拟合范围函数。
            deg (int): chirp 多项式拟合阶数。
            plot (bool): 是否绘图显示拟合的chirp。

        Returns:
            chirp_coeffs (list): chirp 的多项式系数。
        """
        if not hasattr(self, "chirp_data"):
            print(f"Load chirp data first!")
            return None
        data = self.chirp_data
        chirp_delay = [] # 找出每个波长最大值对应的时间
        chirp_wavelength = [] # 不是每个波长的最大值点都有用
        for wl in self.wavelengths:
            delay = float(data[wl].idxmax())
            if check_function(wl, delay):
                chirp_delay.append(delay)
                chirp_wavelength.append(wl)
        self.chirp_coeffs = np.polyfit(chirp_wavelength, chirp_delay, deg=deg)

        if plot:
            plt.scatter(chirp_wavelength, chirp_delay, color='k', s=10)
            plt.plot(self.wavelengths, np.poly1d(self.chirp_coeffs)(self.wavelengths), color='orange', linewidth=2)
            # self.plot_imshow('chirp', 'RdBu_r', 'maxmin', ylim=(np.min(chirp_delay)-1, np.max(chirp_delay)+1))
            plt.show()
        return self.chirp_coeffs

    def calculate_decay(self, delay_cut_start=0.9, deg=3):
        """
        现在考虑，截取一段时间，比如[0.9:]，然后把后面的所有极大值和极小值分别取出来并拟合成上下的边界，用上下边界的平均值作为这段的背景拟合值。拟合可以先考虑多项式拟合。

        Args:
            delay_start (float): 指定延时（单位 ps），自动匹配最接近值。

        Returns:
            decay_constants (dict): 各波长对应的衰减常数（单位 ps）。
        """
        self.decay = {}
        delay_cut_start = get_closest_value(delay_cut_start, self.delays)
        for wl in self.wavelengths:
            data_segment = self.avg_data.loc[delay_cut_start:, wl].dropna()
            if data_segment.empty:
                self.decay[wl] = [None, None, None]
                continue
            peaks, _ = find_peaks(data_segment.values)
            troughs, _ = find_peaks(-data_segment.values)
            peak_delays = data_segment.index.values[peaks]
            peak_values = data_segment.values[peaks]
            trough_delays = data_segment.index.values[troughs]
            trough_values = data_segment.values[troughs]
            if len(peak_delays) < deg or len(trough_delays) < deg:
                self.decay[wl] = [None, None, None]
                continue
            peak_fit_coeffs = np.polyfit(peak_delays, peak_values, deg=deg)
            trough_fit_coeffs = np.polyfit(trough_delays, trough_values, deg=deg)
            self.decay[wl] = [(peak_fit_coeffs + trough_fit_coeffs) / 2, peak_fit_coeffs, trough_fit_coeffs] # 存储背景、上边界、下边界
        return self.decay

        # 旧的指数衰减拟合方案
        # self.exp_params = []
        # for wl in self.wavelengths:
        #     data_segment = self.avg_data.loc[delay_start_cutoff:, wl].dropna()
        #     if data_segment.empty:
        #         self.exp_params.append((np.nan, np.nan, np.nan))
        #         continue
        #     def exp_func(t, A, tau, C):
        #         return A * np.exp(-t / tau) + C
        #     from scipy.optimize import curve_fit
        #     try:
        #         params, _ = curve_fit(exp_func, data_segment.index.values, data_segment.values, p0=(data_segment.values[0], 5, 0))
        #         self.exp_params.append(tuple(params))
        #     except RuntimeError:
        #         self.exp_params.append((np.nan, np.nan, np.nan))
        # print(f'Calculated exponential decay parameters from delay {delay_start_cutoff} ps.')

    def calculate_quantum_beats_old(self,cal_type='fit',cutoff=0.1,f_low = 0.25,f_high = 2.0,width = 0.15,window_length=11,polyorder=3): # 准备删除的旧函数
        """
        通过扣除拟合的指数衰减背景，计算平均数据的量子振荡成分。

        Returns:
            qb_data (pd.DataFrame): 量子振荡成分数据。
        """
        if cal_type == 'fit':
            if not hasattr(self, 'decay'):
                print(f'Calculate exponential decay parameters first!')
                return
            fitted_data = pd.DataFrame(index=self.delays, columns=self.wavelengths)
            fitted_data.index.name = '0' # 保持与原始数据格式一致
            for wl in self.wavelengths:
                bg_coeff = self.decay[wl][0]
                if bg_coeff is None:
                    fitted_data[wl] = np.nan
                else:
                    fitted_data[wl] = np.poly1d(bg_coeff)(self.delays)
            self.qb_data = self.avg_data - fitted_data
            print(f'Calculated quantum beats data.')
        elif cal_type == 'fft_filter': # gpt 版本
            from scipy.fft import fft, ifft, fftfreq
            qb_data = pd.DataFrame(index=self.delays, columns=self.wavelengths,dtype=float)
            qb_data.index.name = '0'

            dt = np.mean(np.diff(self.delays))
            N = len(self.delays)
            freq = fftfreq(N, dt)   # THz if dt is in ps

            # ---- bandpass parameters (based on your plot) ----
                # 低频截止（去背景），单位 THz
                # 高频截止（去噪声）
                # 过渡带宽 (soft edge)

            # ---- soft bandpass filter (tanh) ----
            def soft_edge(x, f0, w):
                return 0.5*(1 + np.tanh((np.abs(x)-f0)/w))

            highpass = soft_edge(freq, f_low, width)          # 去掉低频指数衰减
            lowpass  = 1 - soft_edge(freq, f_high, width)     # 去掉高频噪声
            bandpass = highpass * lowpass
            # plt.plot(freq, bandpass)
            # plt.show()

            for wl in self.wavelengths:
                signal = self.avg_data[wl].values.copy()

                # --- remove coherent artifact (t < 0.5 ps) ---
                mask = self.delays < 0.5
                if np.any(mask):
                    # 用 0.5–1 ps 区域的线性外推替代
                    valid = (~mask) & (self.delays < 1.0)
                    if valid.sum() > 1:
                        p = np.polyfit(self.delays[valid], signal[valid], 1)
                        signal[mask] = np.polyval(p, self.delays[mask])
                    else:
                        signal[mask] = signal[~mask][0]

                signal_fft = fft(signal)

                # --- apply bandpass ---
                filtered_fft = signal_fft * bandpass

                filtered_signal = np.real(ifft(filtered_fft))
                qb_data[wl] = filtered_signal

            self.qb_data = qb_data
            print("Calculated quantum beats data (bandpass FFT).")

        elif cal_type == 'fft_filter_copilot':
            from scipy.fft import fft, ifft, fftfreq
            qb_data = pd.DataFrame(index=self.delays, columns=self.wavelengths,dtype=float)
            qb_data.index.name = '0' # 保持与原始数据格式一致
            dt = np.mean(np.diff(self.delays))
            N = len(self.delays)
            freq = fftfreq(N, dt)
            for wl in self.wavelengths:
                signal = self.avg_data[wl].values
                signal_fft = fft(signal)
                # 设计一个简单的高通滤波器，去除低频成分
                cutoff = cutoff  # 设置截止频率，根据需要调整
                filter_mask = np.abs(freq) > cutoff
                filtered_fft = signal_fft * filter_mask
                filtered_signal = np.real(ifft(filtered_fft))
                qb_data[wl] = filtered_signal
            self.qb_data = qb_data
            print(f'Calculated quantum beats data.')
        elif cal_type == 'savgol':
            from scipy.signal import savgol_filter
            smoothed_data = self.avg_data.apply(lambda col: savgol_filter(col, window_length=window_length, polyorder=polyorder), axis=0)
            self.qb_data = self.avg_data - smoothed_data
            print(f'Calculated quantum beats data.')
            return self.qb_data
        elif cal_type == 'manual':
            m_wl = get_closest_value(485, self.wavelengths)
            m_delay = get_closest_value(1.0, self.delays)
            bg_data = pd.DataFrame(index=self.delays, columns=self.wavelengths, dtype=float)
            bg_data.index.name = '0' # 保持与原始数据格式一致
            # 接下来把选定wl和delay的两列数据矩阵乘法乘出一个背景
            for wl in self.wavelengths:
                for delay in self.delays:
                    bg_data.at[delay, wl] = self.avg_data.at[m_delay, wl] * self.avg_data.at[delay, m_wl] / self.avg_data.at[m_delay, m_wl]
            self.qb_data = self.avg_data - bg_data
            print(f'Calculated quantum beats data.')
            return self.qb_data
        else:
            print(f'Invalid calculation type: {cal_type}.')

    # --------------- Save Data ---------------
    def save_averaged_data(self):
        """
        保存当前平均数据及相关信息。

        生成两个文件：
        - saved_averaged.dat：平均值数据；
        - saved_info.dat：平均圈数、编号、chirp 校正状态。

        Raises:
            ValueError: 若当前未计算平均数据。
        """
        if self.avg_data is None:
            raise ValueError("No averaged data calculated.")

        save_avg_path = os.path.join(self.folder, "saved_averaged.dat")
        self.avg_data.to_csv(save_avg_path, sep='\t')

        # --- 使用 info_mgr 保存元信息 ---
        self.info_mgr.update("data", "averaged_loop_num", self.averaged_loop_num)
        self.info_mgr.update("data", "averaged_loops", self.averaged_loops)
        self.info_mgr.update("chirp", "corrected", getattr(self, "chirp_corrected", False))

        print(f"Saved averaged data → {save_avg_path}")
        print(f"Saved loop info → {self.info_mgr.path}")

    # --------------- Trial functions ---------------
    def remove_scatter_background(self):
        """
        尝试去除散射背景，观察是否效果会变好。
        """
        scatter = self.avg_data.loc[:0.4, :].mean()
        # plt.plot(self.wavelengths, scatter)
        # plt.show()
        self.avg_data = self.avg_data - scatter