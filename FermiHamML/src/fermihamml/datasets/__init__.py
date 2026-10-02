from typing import Dict, Type, Union

from matchcake_opt.datasets import BaseDataset
from matchcake_opt.utils import get_all_subclasses

from .agnews_dataset import AGNewsDataset
from .cifar2_dataset import Cifar2Dataset
from .fashion_dataset import FashionDataset
from .imdb_dataset import IMDbDataset
from .sst2_dataset import SST2Dataset

dataset_name_to_type_map: Dict[str, Type[BaseDataset]] = {
    _cls.DATASET_NAME: _cls
    for _cls in get_all_subclasses(BaseDataset, include_base_cls=False)
    if hasattr(_cls, "DATASET_NAME")
}
dataset_name_to_type_map.update(
    {
        _cls.DATASET_NAME.lower(): _cls
        for _cls in get_all_subclasses(BaseDataset, include_base_cls=False)
        if hasattr(_cls, "DATASET_NAME")
    }
)


def get_dataset_cls_by_name(dataset: Union[str, Type[BaseDataset]]) -> Type[BaseDataset]:  # pragma: no cover
    if isinstance(dataset, str):
        if dataset.lower() not in dataset_name_to_type_map:
            raise ValueError(f"Dataset {dataset} ({dataset.lower()}) not found in {dataset_name_to_type_map.keys()}")
        return dataset_name_to_type_map[dataset.lower()]
    return dataset
