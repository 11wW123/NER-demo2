import json
from pathlib import Path

class Config:
    def __init__(self, config_path):
        config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(self, key, value)

    @property
    def save_dir(self):
        """
        自动生成当前实验的保存目录
        outputs/weibo_bert-base-chinese
        outputs/msra_chinese-bert-wwm
        """
        model_name = self.model_name.split("/")[-1]
        path = Path(self.output_dir) / f"{self.dataset}_{model_name}"
        path.mkdir(parents=True, exist_ok=True)

        return path