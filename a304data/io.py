import numpy as np
import pandas as pd
import os

def load_pp_loop(filename, wl_min=450, wl_max=750):
    """
    从文件中读取 pump-probe 数据，并根据给定波长范围筛选列。

    Args:
        filename (str): 数据文件路径。
        wl_min (float, optional): 波长下限。默认值为 450。
        wl_max (float, optional): 波长上限。默认值为 750。

    Returns:
        pd.DataFrame: 一圈 pump-probe 数据。
    """
    # 现在能够同时支持可见区系统和红外区系统产生的数据格式
    # 可见区的 '0' 列为 delay
    # 红外区的 '0.00000000' 列为 delay
    # 因此通过 pd.to_numeric 转成数值再 set_index(0, inplace=True)
    data = pd.read_table(filename)
    data.columns = pd.to_numeric(data.columns, errors='coerce')
    data.set_index(0, inplace=True)

    # 如果 wl_max 和 wl_min 超过数据范围会报错，所以要限制一下
    wl_max_clipped = min(wl_max, data.columns.max()) 
    wl_min_clipped = max(wl_min, data.columns.min())

    ### 关于数据格式：可见的部分是波长从大到小排，近红外的是从小到大排
    if wl_max < 800: # 判定为可见
        return data.loc[:, wl_max_clipped:wl_min_clipped]
    else:            # 判定为近红外
        return data.loc[:, wl_min_clipped:wl_max_clipped]

def load_uvvis_data(filename):
    """
    从文件中读取 UV-Vis 数据。

    Args:
        filename (str): 数据文件路径。

    Returns:
        pd.DataFrame: 读取的完整数据表，包含'Wavelength'和'Work'。
    """
    data = pd.read_table(filename, sep='\\s+')  # '\\s+'匹配任意空白字符
    return data
    