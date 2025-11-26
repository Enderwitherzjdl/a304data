import os,re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Literal
from .io import load_pp_loop
from .utils import get_closest_value

class PPInfoManager:
    """
    管理 PPLoopDataset 的 metadata 文件（saved_info.dat）。

    支持读取、更新和保存平均值信息、chirp 校正信息等。
    """

    def __init__(self, folder: str):
        """
        初始化并加载 metadata 文件。

        Args:
            folder (str): 数据文件所在目录
        """
        self.path = os.path.join(folder, "saved_info.dat")
        self.info: dict = self._load_info()

    def _load_info(self) -> dict:
        """
        读取 saved_info.dat，如果文件不存在则返回空字典。
        """
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        """
        将当前 info 字典保存到文件。
        """
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=2)

    def update(self, section: str, key: str, value):
        """
        更新 info 中指定 section 的键值，如果 section 不存在会自动创建。

        Args:
            section (str): 信息分类，例如 "data"、"chirp"
            key (str): 键名
            value: 键值
        """
        self.info.setdefault(section, {})
        self.info[section][key] = value
        self.save()

    def get(self, section: str, key: str, default=None):
        """
        获取 info 中指定 section 的键值，如果不存在返回默认值。
        """
        return self.info.get(section, {}).get(key, default)


class PPLoopDataset:
    """
    Pump-Probe 数据集类。
    
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
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.info_mgr = PPInfoManager(folder)
        # 加载数据
        self.data = []
        if read_averaged_only == False:
            self._load_original_data()
        self._load_averaged_data()
        # 获取数据集基本信息
        self.delays = self.data[0].index.values
        self.wavelengths = self.data[0].columns.values

    ########## Load Data ##########
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
        若存在 saved_info.dat 文件，则加载其中的平均圈数信息；
        否则使用当前已加载的圈数作为平均信息。

        Returns:
            bool: 若未找到平均值文件返回 False，否则返回 None。
        """
        avg_file = self._check_saved_averaged_file()
        if avg_file is None:
            print("Averaged file not found.")
            return False
        avg_path = os.path.join(self.folder, avg_file)
        self.avg_data = load_pp_loop(avg_path, self.wl_min, self.wl_max)

        self.averaged_loop_num = self.info_mgr.get("data", "averaged_loop_num", self.loop_num)
        self.averaged_loops = self.info_mgr.get("data", "averaged_loops", list(range(1, self.loop_num+1)))
        self.chirp_corrected = self.info_mgr.get("chirp", "corrected", False)

        print(f'Load averaged file from {avg_path}')
        print(f'Averaged data chirp corrected: {self.chirp_corrected}')

    def _load_original_data(self):
        """
        加载目录中的所有原始 loop 数据文件。

        文件名中需包含 '_loop'。
        使用 load_pp_loop() 读取各文件并保存至 self.data。
        若未检测到任何 loop 数据则报错。

        Raises:
            ValueError: 当目录中未找到任何 loop 数据文件时。
        """
        file_list = [f for f in os.listdir(self.folder) if '_loop' in f]
        for file in file_list:
            full_path = os.path.join(self.folder, file)
            self.data.append(load_pp_loop(full_path, self.wl_min, self.wl_max))
        if not self.data:
            raise ValueError(f"Folder {self.folder} does not contain any loop data.")
        self.loop_num = len(self.data)
        print(f'Load original files: {self.loop_num} loops.')
    
    def load_chirp_data(self,chirp_dir):
        """
        加载 chirp 数据。只允许使用 chirp 数据计算 chirp。

        Args:
            chirp_dir (str): chirp 文件目录，包含有且仅有一个 'averaged' 文件。
        """
        files = [f for f in os.listdir(chirp_dir) if 'averaged' in f]
        if len(files) != 1:
            raise ValueError(f'Not exactly one chirp file.')
        self.chirp_data = load_pp_loop(os.path.join(chirp_dir,files[0]), self.wl_min, self.wl_max)

    ########## Plot Data ##########
    def plot_at_delay(self, delay, plot_list, savefig=False):
        """
        在指定延时处绘制信号曲线。

        Args:
            delay (float): 指定延时（单位 ps），将自动匹配到最接近值。
            plot_list (int | str | list | tuple): 指定绘制内容。
                - 'avg' 表示绘制平均数据；
                - int 表示绘制对应 loop；
                - list/tuple 可包含以上多项。
            savefig (bool, optional): 是否保存图片。默认为 False。

        Raises:
            TypeError: plot_list 类型非法时。
            ValueError: 指定的绘图对象不存在时。
        """
        delay = get_closest_value(delay, self.delays)
        if isinstance(plot_list, (int, str)):
            plot_list = [plot_list]
        elif not isinstance(plot_list, (list, tuple)):
            try:
                plot_list = list(plot_list)
            except TypeError:
                raise TypeError("Invalid plot_list type, plot_list must be int, str, list, tuple or iterable.")
        for item in plot_list:
            if isinstance(item, str) and item.lower() == 'avg':
                if hasattr(self, 'avg_data') and self.avg_data is not None:
                    plt.plot(self.wavelengths, self.avg_data.loc[delay, :], label='Averaged')
                else:
                    raise ValueError("No averaged data calculated.")
            else:
                try:
                    id = int(item)
                except Exception:
                    raise ValueError(f"Invalid plot identifier: {item}")
                if 1 <= id <= self.loop_num:
                    plt.plot(self.wavelengths, self.data[id-1].loc[delay, :], label=f'Loop {id}')
                else:
                    raise ValueError(f"Invalid loop id {id} (should be in [1,{self.loop_num}])")
        plt.xlabel('Wavelength (nm)', fontsize=14)
        plt.ylabel('$\\Delta$O.D.', fontsize=14)
        plt.title(f'Plot at delay {delay:.2f} ps',fontsize=14)
        plt.legend()
        if savefig:
            plt.savefig(os.path.join(self.folder,f'Signal-{delay:.2f}ps.jpg'),bbox_inches='tight',dpi=300)
        plt.show()

    def plot_at_wavelength(self, wl, plot_list, savefig=False):
        """
        在指定波长处绘制信号随延时变化的曲线。

        Args:
            wl (float): 指定波长（单位 nm），将自动匹配到最接近值。
            plot_list (int | str | list | tuple): 指定绘制内容。
                - 'avg' 表示绘制平均数据；
                - int 表示绘制对应 loop；
                - list/tuple 可包含以上多项。
            savefig (bool, optional): 是否保存图片。默认为 False。

        Raises:
            TypeError: plot_list 类型非法时。
            ValueError: 指定的绘图对象不存在时。
        """
        wl = get_closest_value(wl, self.wavelengths)
        if isinstance(plot_list, (int, str)):
            plot_list = [plot_list]
        elif not isinstance(plot_list, (list, tuple)):
            try:
                plot_list = list(plot_list)
            except TypeError:
                raise TypeError("Invalid plot_list type, plot_list must be int, str, list, tuple or iterable.")
        for item in plot_list:
            if isinstance(item, str) and item.lower() == 'avg':
                if hasattr(self, 'avg_data') and self.avg_data is not None:
                    plt.plot(self.delays, self.avg_data.loc[:, wl], label='Averaged')
                else:
                    raise ValueError("No averaged data calculated.")
            else:
                try:
                    id = int(item)
                except Exception:
                    raise ValueError(f"Invalid plot identifier: {item}")
                if 1 <= id <= self.loop_num:
                    plt.plot(self.delays, self.data[id-1].loc[:, wl], label=f'Loop {id}')
                else:
                    raise ValueError(f"Invalid loop id {id} (should be in [1,{self.loop_num}])")
        plt.xlabel('Delay (ps)', fontsize=14)
        plt.ylabel('$\\Delta$O.D.', fontsize=14)
        plt.title(f'Plot at wavelength {wl:.0f} nm',fontsize=14)
        plt.legend()
        if savefig:
            plt.savefig(os.path.join(self.folder,f'Signal-{wl:.0f}nm.jpg'),bbox_inches='tight',dpi=300)
        plt.show()
    
    def plot_with_loop(self, wl, delay, savefig=False):
        """
        绘制指定波长与延时下，各圈信号强度随圈数的变化。

        Args:
            wl (float): 指定波长（单位 nm），自动匹配最接近值。
            delay (float): 指定延时（单位 ps），自动匹配最接近值。
            savefig (bool, optional): 是否保存图片。默认为 False。
        """
        wl = get_closest_value(wl, self.wavelengths)
        delay = get_closest_value(delay, self.delays)
        intensities = []
        for d in self.data:
            intensities.append(d.loc[delay, wl])
        plt.plot(np.arange(1, len(intensities)+1), intensities)
        plt.xlabel('Loop number', fontsize=14)
        plt.ylabel('$\\Delta$O.D.', fontsize=14)
        plt.title(f'Intensity at $\\lambda$={wl:.0f} nm, $\\tau$={delay:.2f} ps, ',fontsize=14)
        if savefig:
            plt.savefig(os.path.join(self.folder,'SignalVsTime-{wl:.0f}nm-{delay:.2f}ps.jpg'),bbox_inches='tight',dpi=300)
        plt.show()

    def plot_imshow(
        self,
        index: Literal['avg'] | None = 'avg',
        cmap: Literal['bwr', 'RdBu_r'] = 'bwr', # 只支持红白蓝配色
        vmaxtype: Literal['maxmin', 'absmax'] = 'maxmin', # 最大值类型
        xlim: tuple = None,
        ylim: tuple = None,
    ):
        """
        热图版本其一：plt.imshow()。

        Args:
            index ('avg' | None): 绘制对象，默认为 'avg'。【以后还要支持'chirp' file】
            cmap (str): 色彩映射，'bwr' 或 'RdBu_r'。
            vmaxtype ('maxmin' | 'absmax'): 最大值类型，'maxmin' 表示取最大值和最小值，'absmax' 表示取绝对值最大值。
            xlim (tuple): x 轴范围，默认为 None。
            ylim (tuple): y 轴范围，默认为 None。
        """
        if index == 'avg': data = self.avg_data
        else: raise ValueError(f"Invalid index: {index}.")
        if vmaxtype == 'maxmin':
            vmax = np.nanmax(data.values); vmin = np.nanmin(data.values)
        elif vmaxtype == 'absmax':
            vmax = np.nanmax(np.abs(data.values)); vmin = -vmax
        else: raise ValueError(f"Invalide vmaxtype: {vmaxtype}.")
        extent = [self.wavelengths[0], self.wavelengths[-1], self.delays[0], self.delays[-1]]
        plt.figure(figsize=(5,4))
        plt.imshow(
            X = data.values,
            aspect = 'auto',        # 'auto' 为拉伸，'equal' 为等比例
            origin = 'lower',
            extent = extent,
            cmap = cmap,
            vmax = vmax,
            vmin = vmin,
        )
        plt.colorbar(label='$\\Delta$O.D.')
        plt.xlabel('Wavelength (nm)',fontsize=14)
        plt.ylabel('Delay (ps)',fontsize=14)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

    ########## Modify Data ##########
    def clean_jump_points(self, ref_wl, threshold=0.02, loop='all'):
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

        Raises:
            ValueError: loop 参数非法时。
        """
        if loop == 'all':
            clean_target = range(self.loop_num)
        else:
            if isinstance(loop, int):
                clean_target = [loop-1]
            elif isinstance(loop, (list, tuple, np.array)):
                clean_target = [l-1 for l in loop]
            else:
                raise ValueError(f"Invalid loop identifier: {loop}")
        wl = get_closest_value(ref_wl, self.wavelengths)
        for id in clean_target:
            data_clean = self.data[id].copy()
            # 第一行和最后一行无法处理
            for i in range(1, len(self.delays)-1):
                curr_val = self.data[id].iloc[i][wl]
                prev_val = self.data[id].iloc[i-1][wl]
                next_val = self.data[id].iloc[i+1][wl]
                if abs(curr_val-prev_val) > threshold and abs(curr_val-next_val) > threshold:
                    data_clean.iloc[i] = (self.data[id].iloc[i-1]+self.data[id].iloc[i+1])/2
                    print(f'Clean jump point in loop {id+1} at {self.delays[i]} ps')
            self.data[id] = data_clean

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
    ) -> list:
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


    def correct_chirp(self):
        """
        对平均数据做 chirp 校正
        """
        if getattr(self, "chirp_corrected", False):
            print(f'Chirp correction has been done before.')
            return
        interpolated_data = pd.DataFrame(index=self.delays, columns=self.wavelengths)
        interpolated_data.index.name = '0' # 保持与原始数据格式一致
        if hasattr(self, 'chirp_coeffs'):
            poly = np.poly1d(self.chirp_coeffs)
        else:
            print(f'Calculate chirp coefficients first!')
            return
        for wl in self.wavelengths:
            interp_func = interp1d(self.delays, self.avg_data[wl], kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data[wl] = interp_func(self.delays + poly(wl) - poly(self.wavelengths[-1]))
        self.avg_data = interpolated_data
        self.chirp_corrected = True
        print(f'Chirp corrected for averaged data.')

    ########## Save Data ##########
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
