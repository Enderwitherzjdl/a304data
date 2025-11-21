import numpy as np

# 根据目标值获取最接近的实际值，适用于找delay和波长
def get_closest_value(target_value, current_array):
    idx = (np.abs(current_array - target_value)).argmin()
    return current_array[idx]
