import os,re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io import load_uvvis_data
from .utils import get_closest_value
from scipy.signal import find_peaks

class UVVisDataset: 
    """
    UVVis 数据集类。

    在初始化时，从指定目录中加载包含 'FX2000_FX2K243001' 标签的数据文件。
    文件名中含有 'bg' 或 'background' 的视为背景文件，其余为光谱文件。
    背景文件应有且仅有一个，光谱文件至少一个。
    """
    def __init__(self, folder):
        """
        初始化 UVVisDataset 实例，并加载原始数据。

        Args:
            folder (str): 数据文件所在的目录路径。
        """
        self.folder = folder
        self._load_original_data()

    ########## Load Data ##########
    def _load_original_data(self):
        """
        扫描目录并加载原始 UVVis 数据。

        在目录中查找文件名包含 'FX2000_FX2K243001' 的文件。
        其中，文件名包含 'bg' 或 'background' 的被视为背景文件，
        其他文件视为光谱文件。

        Raises:
            ValueError: 若未找到或找到多个背景文件，或未找到光谱文件。
        """
        file_list = []
        bg_file_list = []
        for file in os.listdir(self.folder):
            if 'FX2000_FX2K243001' in file:
                if 'bg' in file or 'background' in file:
                    bg_file_list.append(file)
                else:
                    file_list.append(file)
        if len(bg_file_list)!= 1:
            raise ValueError('There should be exactly one background file in the folder')
        if len(file_list) < 1:
            raise ValueError('There should be at least one UVVis data file in the folder')
        self.bg_file = bg_file_list[0]
        self.uvvis_files = file_list
        self.bg_data = load_uvvis_data(os.path.join(self.folder, self.bg_file))
        self.uvvis_data = []
        for file in self.uvvis_files:
            self.uvvis_data.append(load_uvvis_data(os.path.join(self.folder, file)))
        self.uvvis_num = len(self.uvvis_data)
        
    ########## Calculate Data ##########
    def calculate_absorbance(self):
        """
        计算各光谱数据的吸光度，并按最大值归一化。

        吸光度计算公式：
            Absorbance = -log10(Work) + log10(背景数据的 Work)

        结果存储在每个光谱数据的 'Absorbance' 列中。
        """
        for data in self.uvvis_data:
            data['Absorbance'] = -np.log10(data['Work']) + np.log10(self.bg_data['Work'])
        # 按吸光度最大值归一化
        max_abs = max(data['Absorbance'].max() for data in self.uvvis_data)
        for data in self.uvvis_data:
            data['Absorbance'] = data['Absorbance'] / max_abs
         
    def find_peak(self,height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):
        """
        使用 scipy.signal.find_peaks 查找吸光度峰。

        当前仅支持单个光谱数据。
        查找结果的索引存入 self.peaks。

        Args:
            height: 峰的高度阈值。
            threshold: 峰的邻近差值阈值。
            distance: 峰之间的最小距离。
            prominence: 峰的显著性。
            width: 峰宽要求。
            wlen: 计算显著性时使用的窗口长度。
            rel_height: 相对高度参数，用于峰宽计算。
            plateau_size: 平顶宽度要求。

        Raises:
            ValueError: 当数据集中包含多个光谱数据时。
        """
        if self.uvvis_num > 1:
            raise ValueError("Currently find_peak does not support more than one uv-vis data.")
        data = self.uvvis_data[0]  # 取出唯一的数据集
        absorb = data['Absorbance']
        peaks, _ = find_peaks(absorb, height=height, threshold=threshold, 
                            distance=distance, prominence=prominence, width=width, 
                            wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
        self.peaks = list(absorb.index[peaks])

    ########## Plot Data ##########
    def plot_uvvis(self, index=None, wl_min=None, wl_max=None, savefig=False):
        """
        绘制指定光谱数据的吸光度曲线。

        当存在 self.peaks 时，在图中标注峰位。

        Args:
            index (int | list | tuple | None): 指定绘制的数据索引。
                - None: 绘制全部光谱；
                - int: 绘制第 index 个光谱；
                - list 或 tuple: 绘制指定序号的多个光谱。
            wl_min (float | None): x 轴最小波长。
            wl_max (float | None): x 轴最大波长。
            savefig (bool): 是否保存图像文件。

        Raises:
            TypeError: index 类型不符合要求时。
        """
        if index is None:
            index = range(1,len(self.uvvis_data)+1)
        elif isinstance(index, int):
            index = [index]
        else:
            try:
                index = list(index)
            except TypeError:
                raise TypeError('Index should be int, list or tuple.')
        for id in index:
            plt.plot(self.bg_data['Wavelength'], self.uvvis_data[id-1]['Absorbance'])
        if wl_min is None: wl_min = self.bg_data['Wavelength'].min()
        if wl_max is None: wl_max = self.bg_data['Wavelength'].max()
        # 标峰
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            data = self.uvvis_data[0]  # 取出唯一的数据集
            coeff_left = 0.105; coeff_right = 0.01; coeff_len = 0.095
            for peak_index in self.peaks:
                plt.vlines(data['Wavelength'].iloc[peak_index], data['Absorbance'].iloc[peak_index]+0.01, 1.05,'k')
                # 标签默认在右边
                text_wl = data['Wavelength'].iloc[peak_index] + coeff_right*(wl_max - wl_min)
                # 右边出框了则改到左边
                if text_wl + coeff_len*(wl_max - wl_min) > wl_max:
                   text_wl = data['Wavelength'].iloc[peak_index] - coeff_left*(wl_max - wl_min) 
                text_abs = 1.02
                plt.text(text_wl, text_abs, str(data['Wavelength'].iloc[peak_index]), color='k')
        plt.xlim(wl_min, wl_max)
        plt.xlabel('Wavelength (nm)', fontsize=14)
        plt.ylabel('Absorbance', fontsize=14)
        plt.title('UVVis Spectrum', fontsize=14)
        if savefig:
            plt.savefig(os.path.join(self.folder, 'uvvis_spectrum.png'))
        plt.show()

    def plot_background(self, wl_min=None, wl_max=None, savefig=False):
        """
        绘制背景数据的工作曲线。

        Args:
            wl_min (float | None): x 轴最小波长。
            wl_max (float | None): x 轴最大波长。
            savefig (bool): 是否保存图像文件。
        """
        if wl_min is None: wl_min = self.bg_data['Wavelength'].min()
        if wl_max is None: wl_max = self.bg_data['Wavelength'].max()
        plt.plot(self.bg_data['Wavelength'], self.bg_data['Work'])
        plt.xlim(wl_min, wl_max)
        plt.xlabel('Wavelength (nm)', fontsize=14)
        plt.ylabel('Work', fontsize=14)
        plt.title('Background', fontsize=14)
        if savefig:
            plt.savefig(os.path.join(self.folder, 'background_work_curve.png'))
        plt.show()
        