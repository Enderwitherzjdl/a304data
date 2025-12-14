# a304data 包

> 快速处理 CCME A304 的各种数据，进行数据分析

---

##  安装（Installation）

目前尚未形成稳定版本，需要的话可以以开发者模式安装

```bash
git clone https://github.com/yourname/a304data.git
cd a304data
pip install -e .
```

---

## 快速开始（Quick Start）

```python
# 以最常用的 pp 数据处理为例
from a304data import PPLoopDataset

ds = PPLoopDataset('.' ,450, 750) # 指定目录和波长范围
ds.calculate_averaged_data() # 从原始数据计算平均值
ds.correct.chirp('/your/chirp/dir') # 啁啾校正（可见区）
ds.correct.delay_zero(0.35) # 零点校正（只影响作图）

ds.plot.at_wavelength([490,495,500], 'avg',savefig=True) # 作图并保存
ds.plot.at_delay(2, ['avg',1,2,3])

```

> 说明：这里给一个**最小可运行示例**，展示典型使用流程。

---

## 项目结构（Project Structure）

```text
a304data
├── correct/          # pp数据校正
├── plot/             # pp作图
├── qbanalyzer/       # pp的qb分析
├── __init__.py
├── info_manager.py   # pp数据信息管理
├── io.py             # 数据读写
├── pploopdataset.py  # pp数据结构
├── utils.py          # 通用工具函数
└── uvvisdataset.py   # uvvis数据结构
```

---

## 文档（Documentation）

* API 文档：见 `docs/` (TODO)
* 示例脚本：见 `examples/` (TODO)

---

## 贡献（Contributing）

1. Fork 本仓库
2. 新建分支：`feature/xxx`
3. 提交代码并写清 commit message
4. 提交 Pull Request

---

## 许可证（License）

MIT License
