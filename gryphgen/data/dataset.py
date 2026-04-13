from pathlib import Path
from typing import List

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.registry import DATASETS


@DATASETS.register_module()
class FormulaDataset(BaseDataset):
    def load_data_list(self) -> List[dict]:
        # load preprocessed pickle
        paths = Path(self.ann_file).expanduser().rglob("*.pkl")
        return list(load(path, file_format="pickle") for path in paths)


@DATASETS.register_module()
class FormulaPathDataset(BaseDataset):
    def load_data_list(self):
        paths = Path(self.ann_file).expanduser().rglob("*.pkl")
        return list(dict(pkl_path=str(p)) for p in paths)


@DATASETS.register_module()
class FormulaEditDataset(BaseDataset):
    @property
    def edit(self):
        return self.filter_cfg.get("edit")

    @property
    def split(self):
        return self.filter_cfg.get("split")

    def load_data_list(self) -> List[dict]:
        path = Path(self.ann_file).expanduser()
        data = load(path, file_format="pickle")
        split_dict = data[self.edit][self.split]

        data_list = []
        for sample_id, rec in split_dict.items():
            item = dict(rec)
            item["sample_id"] = sample_id
            data_list.append(item)

        return data_list
