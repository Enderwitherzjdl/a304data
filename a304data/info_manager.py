import os, re, json
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
