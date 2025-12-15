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

    def _set_plot_style(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        fontsize: int = 14,
        legend: bool = True,
    ):
        if title:
            plt.title(title, fontsize=fontsize)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlabel:
            plt.xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            plt.ylabel(ylabel, fontsize=fontsize)
        if legend:
            plt.legend()

    def _get_symbol(self, datatype:str):
        """
        Get plot and naming symbols.

        Args:
            datatype (str): self.ds.type, 'VIS' or 'IR' or 'wavelength' or 'wavenumber'
        
        Returns:
            symbols (dict): coord,Coord,unit,unitplain,symbol
        """
        if datatype == 'VIS' or datatype == 'wavelength':
            return {'coord':'wavelength', 'Coord':'Wavelength', 'unit':'nm', 'unitplain':'nm', 'symbol':r'$\\lambda$'}
        elif datatype == 'IR' or datatype == 'wavenumber':
            return {'coord':'wavenumber', 'Coord':'Wavenumber', 'unit':r'cm$^{-1}$', 'unitplain':'cm-1', 'symbol':r'\\tilde{\\nu}'}
        else:
            return None

    def _get_vmin_vmax(
        self, 
        data, 
        vmaxtype: Literal['maxmin', 'absmax'] = 'maxmin',
        vlim: float | tuple[float, float] = None,
    ) -> tuple[float, float]:
        if vlim is not None:
            if isinstance(vlim, (int, float)):
                vmin, vmax = -abs(vlim), abs(vlim)
            else:
                vmin, vmax = vlim
        else:
            if vmaxtype == 'maxmin':
                vmax = np.nanmax(data); vmin = np.nanmin(data)
            elif vmaxtype == 'absmax':
                vmax = np.nanmax(np.abs(data)); vmin = -vmax
            else: raise ValueError(f"Invalide vmaxtype: {vmaxtype}.") 
        return vmin, vmax       

    def at_delay(
        self, 
        delay, 
        plot_list, 
        savefig = False,
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
        
        sym = self._get_symbol(self.ds.type)
        self._set_plot_style(
            title = f'Signal at {delay:.2f} ps' if isinstance(delay,(int, float)) else f'Signal at selected delays' ,
            xlabel = f'{sym['Coord']} ({sym['unit']})' ,
            ylabel = '$\\Delta$O.D.',
            xlim = xlim,
            ylim = ylim,
        )
        if savefig:
            if isinstance(delay, (int, float)):
                plt.savefig(os.path.join(self.ds.folder, f'Signal-{delay:.2f}ps.jpg'),bbox_inches='tight',dpi=300)
            else:
                plt.savefig(os.path.join(self.ds.folder, f'Singal-selected_delays.jpg'), bbox_inches='tight', dpi=300)
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
        在指定波长/波数处绘制信号随延时变化的曲线。还支持绘制拟合的背景曲线。

        Args:
            wl (float | list | tuple): 指定波长/波数（单位 nm/cm-1），将自动匹配到最接近值。
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
        
        sym = self._get_symbol(self.ds.type)

        if isinstance(wl, (list, tuple)):
            for w in wl:
                w = get_closest_value(w, self.ds.wavelengths)
                if plot_list == 'avg':
                    plt.plot(self.ds.delays, self.ds.avg_data.loc[:, w], label=f'{w:.0f} {sym['unit']}')
                else:
                    plt.plot(self.ds.delays, self.ds.data[id-1].loc[:, w], label=f'{w:.0f} {sym['unit']}')
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
                    
        self._set_plot_style(
            title = f'Signal at {sym['coord']} {wl:.0f} {sym['unit']}' if isinstance(wl,(int, float)) else f'Signal at selected {sym['coord']}s' ,
            xlabel = 'Delay (ps)',
            ylabel = '$\\Delta$O.D.',
            xlim = xlim,
            ylim = ylim,
        )
        if savefig:
            if isinstance(wl, (int, float)):
                plt.savefig(os.path.join(self.ds.folder,f'Signal-{wl:.0f}{sym['unitplain']}.jpg'),bbox_inches='tight',dpi=300)
            else:
                plt.savefig(os.path.join(self.ds.folder, f'Singal-selected_{sym['coord']}s.jpg'), bbox_inches='tight', dpi=300)
        plt.show()
    
    def with_loop(self, wl, delay, savefig=False, xlim=None, ylim=None):
        """
        绘制指定波长/波数与延时下，各圈信号强度随圈数的变化。

        Args:
            wl (float): 指定波长/波数（单位 nm/cm-1），自动匹配最接近值。
            delay (float): 指定延时（单位 ps），自动匹配最接近值。
            savefig (bool, optional): 是否保存图片。默认为 False。
        """
        wl = get_closest_value(wl, self.ds.wavelengths)
        delay = get_closest_value(delay, self.ds.delays)
        intensities = []
        for d in self.ds.data:
            intensities.append(d.loc[delay, wl])
        plt.plot(np.arange(1, len(intensities)+1), intensities)
        
        sym = self._get_symbol(self.ds.type)
        self._set_plot_style(
            title = f'Intensity at {sym['symbol']}={wl:.0f} {sym['unit']}, $\\tau$={delay:.2f} ps',
            xlabel = 'Loop number',
            ylabel = '$\\Delta$O.D.',
            xlim = xlim,
            ylim = ylim,
        )
        if savefig:
            plt.savefig(os.path.join(self.ds.folder,f'SignalVsTime-{wl:.0f}{sym['unitplain']}-{delay:.2f}ps.jpg'),bbox_inches='tight',dpi=300)
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
        使用 plt.imshow() 绘制热图。

        Args:
            index ('avg','qb' | None): 绘制对象，默认为 'avg'。
            cmap (str): 色彩映射，'bwr' 或 'RdBu_r'。
            vmaxtype ('maxmin' | 'absmax'): 最大值类型，'maxmin' 表示取最大值和最小值，'absmax' 表示取绝对值最大值。
            vlim (float | tuple[flaot,float] | None): 自定义 (vmax, vmin) 并无视 vmaxtype，默认为 None。
            xlim (tuple): x 轴范围，默认为 None。
            ylim (tuple): y 轴范围，默认为 None。
        """
        if index == 'avg': data = self.ds.avg_data.values
        elif index == 'qb': data = self.ds.qb_data.values
        else: raise ValueError(f"Invalid index: {index}.")
        vmin, vmax = self._get_vmin_vmax(data, vmaxtype, vlim)
        extent = [self.ds.wavelengths[0], self.ds.wavelengths[-1], self.ds.delays[0], self.ds.delays[-1]]
        plt.figure(figsize=(5,4))
        plt.imshow(
            X = data,
            aspect = 'auto',        # 'auto' 为拉伸，'equal' 为等比例
            origin = 'lower',
            extent = extent,
            cmap = cmap,
            vmax = vmax,
            vmin = vmin,
        )
        plt.colorbar(label='$\\Delta$O.D.')
        sym = self._get_symbol(self.ds.type)
        self._set_plot_style(
            title = f'{self.ds.pump_wl} nm pump'+f' ({self.ds.qb_method})'if index=='qb' else None,
            xlabel = f'{sym['Coord']} ({sym['unit']})',
            ylabel = 'Delay (ps)',
            xlim = xlim,
            ylim = ylim,
            legend = False,
        )
        plt.show()

    def imshow_freq(
        self,
        index: Literal['abs','angle','real','imag'] | None = 'abs',
        cmap: Literal['bwr', 'RdBu_r'] = 'bwr', # 只支持红白蓝配色
        vmaxtype: Literal['maxmin', 'absmax'] = 'maxmin', # 最大值类型
        vlim: float | tuple[float, float] = None,
        xlim: tuple = None,
        ylim: tuple = None,
    ):
        if not hasattr(self.ds, 'fft_data'):
            raise ValueError(f'Use ds.qb.fft() calculate fft_data first!')
        if index == 'abs':
            data = np.abs(self.ds.fft_data.values)
        elif index == 'angle':
            data = np.angle(self.ds.fft_data.values)
        elif index == 'real':
            data = np.real(self.ds.fft_data.values)
        elif index == 'imag':
            data = np.imag(self.ds.fft_data.values)
        
        vmin, vmax = self._get_vmin_vmax(data, vmaxtype, vlim)
        extent = [self.ds.wavelengths[0], self.ds.wavelengths[-1], self.ds.freq[0], self.ds.freq[-1]]
        plt.figure(figsize=(5,4))
        plt.imshow(
            X = data,
            aspect = 'auto',        # 'auto' 为拉伸，'equal' 为等比例
            origin = 'lower',
            extent = extent,
            cmap = cmap,
            vmax = vmax,
            vmin = vmin,
        )
        plt.colorbar(label='FFT Intensity')
        sym_wn = self._get_symbol('wavenumber')
        sym_wl = self._get_symbol('wavelength')
        self._set_plot_style(
            title = f'QB of {self.ds.pump_wl} nm pump ({self.ds.qb_method})',
            xlabel = f'{sym_wl['Coord']} ({sym_wl['unit']})',
            ylabel = f'{sym_wn['Coord']} ({sym_wn['unit']})',
            xlim = xlim,
            ylim = ylim,
            legend = False,
        )
        plt.show()        