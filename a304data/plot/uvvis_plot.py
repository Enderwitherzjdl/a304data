# a304data/plot/uvvis_plot.py
import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils import get_closest_value

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a304data.uvvisdataset import UVVisDataset

class UVVisPlotTool:
    def __init__(self, ds):
        self.ds = ds

# TODO 把旧代码迁移过来