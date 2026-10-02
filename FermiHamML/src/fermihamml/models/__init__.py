from typing import Dict, Type, Union

from matchcake_opt.modules.base_model import BaseModel
from matchcake_opt.utils import get_all_subclasses

from .nif_ising_cls import NIFIsingCls

models_name_to_type_map: Dict[str, Type[BaseModel]] = {
    _cls.MODEL_NAME: _cls
    for _cls in get_all_subclasses(BaseModel, include_base_cls=False)
    if hasattr(_cls, "MODEL_NAME")
}
models_name_to_type_map.update(
    {
        _cls.MODEL_NAME.lower(): _cls
        for _cls in get_all_subclasses(BaseModel, include_base_cls=False)
        if hasattr(_cls, "MODEL_NAME")
    }
)


def get_model_cls_by_name(model: Union[str, Type[BaseModel]]) -> Type[BaseModel]:  # pragma: no cover
    if isinstance(model, str):
        if model.lower() not in models_name_to_type_map:
            raise ValueError(f"Model {model} ({model.lower()}) not found in {models_name_to_type_map.keys()}")
        return models_name_to_type_map[model.lower()]
    return model
