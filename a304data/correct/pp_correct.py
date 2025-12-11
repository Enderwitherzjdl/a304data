# a304data/plot/pp_correct.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from scipy.interpolate import interp1d
from ..utils import get_closest_value

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a304data.pploopdataset import PPLoopDataset

class PPCorrectTool:
    def __init__(self, ds:"PPLoopDataset"):
        self.ds = ds
    
    def chirp(self, chirp_dir:str, check_function=None, deg=3, plot=False):
        """
        校正数据中的 chirp。

        Args:
            chirp_dir (str): chirp 文件目录，包含有且仅有一个 'averaged' 文件。
            check_function (lambda): 自定义chirp拟合范围函数，配置了一个默认函数。
            deg (int): chirp 多项式拟合阶数。
            plot (bool): 是否绘图显示拟合的chirp。

        Returns:
            chirp_coeffs (list): chirp 的多项式系数。
        """
        if getattr(self.ds, "chirp_corrected", False):
            print(f'Chirp correction has been done before.')
            return
        self.ds._load_chirp_data(chirp_dir)
        def default_check_function(wavelength, delay):
            # 适配'~/chirp/20251025-pump-530nm-8uW-2500ps-chirp'的chirp区域检定函数
            if delay > 0.9 or delay < -0.8:
                return False
            if wavelength > 620 and delay < 0.6:
                return False
            if wavelength > 650 and delay < 0.7:
                return False
            return True
        if check_function is None:
            check_function = default_check_function
        self.ds.calculate_chirp(check_function, deg, plot)
        print(f'Correcting chirp with coefficients: {self.ds.chirp_coeffs}')
        interpolated_data = pd.DataFrame(index=self.ds.delays, columns=self.ds.wavelengths)
        interpolated_data.index.name = '0' # 保持与原始数据格式一致
        if hasattr(self.ds, 'chirp_coeffs'):
            poly = np.poly1d(self.ds.chirp_coeffs)
        else:
            print(f'Calculate chirp coefficients first!')
            return
        for wl in self.ds.wavelengths:
            interp_func = interp1d(self.ds.delays, self.ds.avg_data[wl], kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data[wl] = interp_func(self.ds.delays + poly(wl) - poly(self.ds.wavelengths[-1]))
        self.ds.avg_data = interpolated_data
        self.ds.chirp_corrected = True
        print(f'Chirp corrected for averaged data.')

    def jump_points(self, ref_wl, threshold=0.02, loop='all',delay_range=None):
        """
        基于指定的波长，清除跳点/坏点。

        对每个圈的指定波长数据逐点检查，
        若该点与前后两点的差值均超过阈值 threshold，
        则用相邻两点平均值替换。

        Args:
            ref_wl (float): 参考波长，用于判断跳点。
            threshold (float, optional): 判定阈值，默认 0.02。
            loop (int | list | tuple | str): 指定处理的圈，
                - 'all' 表示全部圈；
                - int 或序列表示指定圈。
            delay_range (tuple | None): 指定延时范围 (min_delay, max_delay)，仅在该范围内清除跳点。默认为 None，表示全范围处理。

        Raises:
            ValueError: loop 参数非法时。
        """
        if loop == 'all':
            clean_target = range(self.ds.loop_num)
        else:
            if isinstance(loop, int):
                clean_target = [loop-1]
            elif isinstance(loop, (list, tuple, np.array)):
                clean_target = [l-1 for l in loop]
            else:
                raise ValueError(f"Invalid loop identifier: {loop}")
        wl = get_closest_value(ref_wl, self.ds.wavelengths)
        count = 0
        for id in clean_target:
            data_clean = self.ds.data[id].copy()
            # 第一行和最后一行无法处理
            for i in range(1, len(self.ds.delays)-1):
                if delay_range is not None:
                    if not (delay_range[0] <= self.ds.delays[i] <= delay_range[1]):
                        continue
                curr_val = self.ds.data[id].iloc[i][wl]
                prev_val = self.ds.data[id].iloc[i-1][wl]
                next_val = self.ds.data[id].iloc[i+1][wl]
                if curr_val-prev_val > threshold and curr_val-next_val > threshold:  # 跳点似乎都是变大。如果采用绝对值检测，会导致相邻三个点跳了两个的情况出问题
                # if abs(curr_val-prev_val) > threshold and abs(curr_val-next_val) > threshold:
                    data_clean.iloc[i] = (self.ds.data[id].iloc[i-1]+self.ds.data[id].iloc[i+1])/2
                    print(f'Clean jump point in loop {id+1} at {self.ds.delays[i]} ps')
                    count += 1
            self.ds.data[id] = data_clean
        print(f'Cleaned {count} jump points in total.')
