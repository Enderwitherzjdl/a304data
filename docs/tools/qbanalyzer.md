## a304data.qbanalyzer

### 模块简介

`qbanalyzer` 模块用于对 **pump–probe (PP) 数据中的量子拍（Quantum Beats, QB）信号**进行分析。该模块不直接存储数据，而是作为 **分析工具（Tool）** 挂载在 `PPLoopDataset` 实例上。

```python
ds.qb
```

---

## 设计思想

* **Dataset-centered**：
  `QBAnalyzer` 不持有原始数据副本，而是通过引用 `PPLoopDataset` 来访问：

  * `self.ds.data / self.ds.avg_data`
  * `self.ds.delays`
  * `self.ds.wavelengths`

* **分析与数据解耦**：
  量子拍分析逻辑不写入 `PPLoopDataset`，避免数据结构类膨胀。

* **状态最小化**：
  除非必要，`QBAnalyzer` 本身不缓存中间结果，分析结果以返回值形式给出。

---

## 核心类

### `QBAnalyzer`

```python
QBAnalyzer(dataset: PPLoopDataset)
```

**参数**：

* `dataset`：`PPLoopDataset` 实例

**典型访问方式**：

```python
ds = PPLoopDataset(...)
ds.qb.some_method(...)
```

---

## 主要分析方法

类中实现了几种量子拍的提取方法，使用时应确认不同方法能给出几乎一致的结果，以保证结果的准确性。

### `savgol(window_length, polyorder)`

**用途**：
使用 Savitzky-Golay 滤波拟合背景并提取量子拍。

**参数**：

* `window_length`：滤波窗口长度。
* `polyorder`：滤波使用的多项式阶数。

**缺陷**：
* Artifact 附近明显受影响，有失真，但不影响信号主体分析。
* delay 的末端有一段无法被拟合。

---

### `poly(delay_cutoff, deg)`

**用途**：
使用多项式拟合背景并提取量子拍。
$$
BG(\lambda,\tau) = \sum_{n=0}^{deg} b_n(\lambda)\tau^n
$$

**参数**：

* `delay_cutoff`：拟合的起始 delay，应避开 artifact。
* `deg`：多项式阶数，需要阶数足够高才能有较好拟合效果，例如取 8~16。

**缺陷**：
* 多项式阶数太低时，容易引入额外的低频背景。太高可能引起拟合时 warning。

---

### `exp(delay_cutoff, n_exp)`

**用途**：
使用多指数衰减拟合背景并提取量子拍。
$$
BG(\lambda,\tau) = \sum_{i=1}^{n_{exp}} A_i(\lambda) e^{-\tau/t_i(\lambda)} + C(\lambda)
$$
参数初猜对拟合结果有明显影响。目前策略是从待拟合曲线暴力猜测参数，以及从临近波长获取拟合参数作为初猜，取结果更好的一个。

**参数**：

* `delay_cutoff`：拟合的起始 delay，应避开 artifact。
* `n_exp`：指数项数，建议取2。取1时经常不能干净地拟合背景，大于等于3时通常物理上不可信。

**缺陷**：
* 拟合结果受初猜、delay_cutoff 影响，需要精细调控。

---


## 典型使用流程

```python
# 构建数据集
ds = PPLoopDataset(folder, wl_min=450, wl_max=650)
# 计算qb
ds.qb.exp(delay_cutoff=0, n_exp=2)
```

---

## 相关模块

* `a304data.pploopdataset`：数据结构定义
* `a304data.correct`：背景与信号校正
* `a304data.plot`：QB 结果可视化
