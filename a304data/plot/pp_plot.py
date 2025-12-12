# a304data/plot/pp_plot.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from ..utils import get_closest_value

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a304data.pploopdataset import PPLoopDataset

class PPPlotTool:
    def __init__(self, ds:"PPLoopDataset"):
        self.ds = ds

    def at_delay(
        self, 
        delay, 
        plot_list, 
        savefig=False,
        xlim: tuple = None,
        ylim: tuple = None,
        cmap: str = 'viridis'
    ):
        """
        在指定延时处绘制信号曲线。

        Args:
            delay (float | list | tuple): 指定延时（单位 ps），将自动匹配到最接近值。
            plot_list (int | str | list | tuple): 指定绘制内容。
                - 'avg' 表示绘制平均数据；
                - int 表示绘制对应 loop；
                - list/tuple 可包含以上多项。
            savefig (bool, optional): 是否保存图片。默认为 False。

        Raises:
            TypeError: plot_list 类型非法时。
            ValueError: 指定的绘图对象不存在时。
        """
        if isinstance(delay, (list, tuple)) and isinstance(plot_list, (list, tuple)):
            raise ValueError(f'目前不支持 delay 和 plot_list 同时为列表,请分开绘图。')
        if isinstance(delay, (list, tuple)):
            # 先把 delay 全部贴最近值
            delays = [get_closest_value(d, self.ds.delays) for d in delay]

            use_color_map = len(delays) >= 5
            if use_color_map:
                # 使用渐变色（示例：viridis，你可换其它 cmap）
                cmap = plt.get_cmap(cmap)
                colors = [cmap(i / (len(delays) - 1)) for i in range(len(delays))]
            else:
                colors = [None] * len(delays)  # 使用默认颜色

            for d, c in zip(delays, colors):
                if plot_list == 'avg':
                    plt.plot(
                        self.ds.wavelengths,
                        self.ds.avg_data.loc[d, :],
                        label=f'{d:.1f} ps',
                        color=c
                    )
                else:
                    plt.plot(
                        self.ds.wavelengths,
                        self.ds.data[id-1].loc[d, :],
                        label=f'{d:.1f} ps',
                        color=c
                    )
        else:
            delay = get_closest_value(delay, self.ds.delays)
            if isinstance(plot_list, (int, str)):
                plot_list = [plot_list]
            elif not isinstance(plot_list, (list, tuple)):
                try:
                    plot_list = list(plot_list)
                except TypeError:
                    raise TypeError("Invalid plot_list type, plot_list must be int, str, list, tuple or iterable.")
            for item in plot_list:
                if isinstance(item, str) and item.lower() == 'avg':
                    if hasattr(self.ds, 'avg_data') and self.ds.avg_data is not None:
                        plt.plot(self.ds.wavelengths, self.ds.avg_data.loc[delay, :], label='Averaged')
                    else:
                        raise ValueError("No averaged data calculated.")
                else:
                    try:
                        id = int(item)
                    except Exception:
                        raise ValueError(f"Invalid plot identifier: {item}")
                    if 1 <= id <= self.ds.loop_num:
                        plt.plot(self.ds.wavelengths, self.ds.data[id-1].loc[delay, :], label=f'Loop {id}')
                    else:
                        raise ValueError(f"Invalid loop id {id} (should be in [1,{self.ds.loop_num}])")
        plt.xlabel('Wavelength (nm)', fontsize=14)
        plt.ylabel('$\\Delta$O.D.', fontsize=14)
        if isinstance(delay, (list, tuple)):
            plt.title(f'Plot loop:{plot_list} at selected delays',fontsize=14)
        else:
            plt.title(f'Plot at delay {delay:.2f} ps',fontsize=14)
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        if savefig:
            plt.savefig(os.path.join(self.ds.folder,f'Signal-{delay:.2f}ps.jpg'),bbox_inches='tight',dpi=300)
        plt.show()

    def at_wavelength(
        self, 
        wl, 
        plot_list, 
        savefig=False,
        xlim: tuple = None,
        ylim: tuple = None,
    ):
        """
        在指定波长处绘制信号随延时变化的曲线。还支持绘制拟合的指数衰减曲线。

        Args:
            wl (float | list | tuple): 指定波长（单位 nm），将自动匹配到最接近值。
            plot_list (int | str | list | tuple): 指定绘制内容。
                - 'avg' 表示绘制平均数据；
                - 'bg' 表示绘制拟合的衰减背景（需调用 .qb 中的方法）；
                - int 表示绘制对应 loop；
                - list/tuple 可包含以上多项。
            savefig (bool, optional): 是否保存图片。默认为 False。

        Raises:
            TypeError: plot_list 类型非法时。
            ValueError: 指定的绘图对象不存在时。
        """
        if isinstance(wl, (list, tuple)) and isinstance(plot_list, (list, tuple)):
            raise ValueError(f'目前不支持 delay 和 plot_list 同时为列表,请分开绘图。')
        
        if isinstance(wl, (list, tuple)):
            for w in wl:
                w = get_closest_value(w, self.ds.wavelengths)
                if plot_list == 'avg':
                    plt.plot(self.ds.delays, self.ds.avg_data.loc[:, w], label=f'{w:.0f} nm')
                else:
                    plt.plot(self.ds.delays, self.ds.data[id-1].loc[:, w], label=f'{w:.0f} nm')
        else:
            wl = get_closest_value(wl, self.ds.wavelengths)
            if isinstance(plot_list, (int, str)):
                plot_list = [plot_list]
            elif not isinstance(plot_list, (list, tuple)):
                try:
                    plot_list = list(plot_list)
                except TypeError:
                    raise TypeError("Invalid plot_list type, plot_list must be int, str, list, tuple or iterable.")
            for item in plot_list:
                if isinstance(item, str):
                    if item.lower() == 'avg':
                        if hasattr(self.ds, 'avg_data') and self.ds.avg_data is not None:
                            plt.plot(self.ds.delays, self.ds.avg_data.loc[:, wl], label='Averaged')
                        else:
                            raise ValueError("No averaged data calculated.")
                    elif item.lower() == 'bg':
                        if hasattr(self.ds, 'bg_data'):
                            plt.plot(self.ds.delays, self.ds.bg_data.loc[:, wl], label='Background')
                        else:
                            raise ValueError("No background data calculated.")
                else:
                    try:
                        id = int(item)
                    except Exception:
                        raise ValueError(f"Invalid plot identifier: {item}")
                    if 1 <= id <= self.ds.loop_num:
                        plt.plot(self.ds.delays, self.ds.data[id-1].loc[:, wl], label=f'Loop {id}')
                    else:
                        raise ValueError(f"Invalid loop id {id} (should be in [1,{self.ds.loop_num}])")
        plt.xlabel('Delay (ps)', fontsize=14)
        plt.ylabel('$\\Delta$O.D.', fontsize=14)
        if isinstance(wl, (list, tuple)):
            plt.title(f'Plot loop:{plot_list} at selected wavelengths',fontsize=14)
        else:
            plt.title(f'Plot at wavelength {wl:.0f} nm',fontsize=14)
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        if savefig:
            plt.savefig(os.path.join(self.ds.folder,f'Signal-{wl:.0f}nm.jpg'),bbox_inches='tight',dpi=300)
        plt.show()
    
    def with_loop(self, wl, delay, savefig=False):
        """
        绘制指定波长与延时下，各圈信号强度随圈数的变化。

        Args:
            wl (float): 指定波长（单位 nm），自动匹配最接近值。
            delay (float): 指定延时（单位 ps），自动匹配最接近值。
            savefig (bool, optional): 是否保存图片。默认为 False。
        """
        wl = get_closest_value(wl, self.ds.wavelengths)
        delay = get_closest_value(delay, self.ds.delays)
        intensities = []
        for d in self.ds.data:
            intensities.append(d.loc[delay, wl])
        plt.plot(np.arange(1, len(intensities)+1), intensities)
        plt.xlabel('Loop number', fontsize=14)
        plt.ylabel('$\\Delta$O.D.', fontsize=14)
        plt.title(f'Intensity at $\\lambda$={wl:.0f} nm, $\\tau$={delay:.2f} ps, ',fontsize=14)
        if savefig:
            plt.savefig(os.path.join(self.ds.folder,'SignalVsTime-{wl:.0f}nm-{delay:.2f}ps.jpg'),bbox_inches='tight',dpi=300)
        plt.show()

    def imshow(
        self,
        index: Literal['avg','qb'] | None = 'avg',
        cmap: Literal['bwr', 'RdBu_r'] = 'bwr', # 只支持红白蓝配色
        vmaxtype: Literal['maxmin', 'absmax'] = 'maxmin', # 最大值类型
        vlim: float | tuple[float, float] = None,
        xlim: tuple = None,
        ylim: tuple = None,
    ):
        """
        热图版本其一：plt.imshow()。

        Args:
            index ('avg','qb' | None): 绘制对象，默认为 'avg'。【以后还要支持'chirp' file】
            cmap (str): 色彩映射，'bwr' 或 'RdBu_r'。
            vmaxtype ('maxmin' | 'absmax'): 最大值类型，'maxmin' 表示取最大值和最小值，'absmax' 表示取绝对值最大值。
            vlim (float | tuple[flaot,float] | None): 自定义 (vmax, vmin) 并无视 vmaxtype，默认为 None。
            xlim (tuple): x 轴范围，默认为 None。
            ylim (tuple): y 轴范围，默认为 None。
        """
        if index == 'avg': data = self.ds.avg_data
        elif index == 'qb': data = self.ds.qb_data
        else: raise ValueError(f"Invalid index: {index}.")
        if vlim is not None:
            if isinstance(vlim, (int, float)):
                vmin, vmax = -abs(vlim), abs(vlim)
            else:
                vmin, vmax = vlim
        else:
            if vmaxtype == 'maxmin':
                vmax = np.nanmax(data.values); vmin = np.nanmin(data.values)
            elif vmaxtype == 'absmax':
                vmax = np.nanmax(np.abs(data.values)); vmin = -vmax
            else: raise ValueError(f"Invalide vmaxtype: {vmaxtype}.")
        extent = [self.ds.wavelengths[0], self.ds.wavelengths[-1], self.ds.delays[0], self.ds.delays[-1]]
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

