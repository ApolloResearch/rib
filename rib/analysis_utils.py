from typing import Any, Iterable

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from rib.data_accumulator import run_dataset_through_model
from rib.hook_fns import rotated_acts_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.interaction_algos import InteractionRotation
from rib.models.utils import get_model_attr


def parse_c_infos(c_infos: list[dict[str, Any]]) -> dict[str, InteractionRotation]:
    return {c_info["node_layer_name"]: InteractionRotation(**c_info) for c_info in c_infos}


def get_rib_acts(
    hooked_model: HookedModel,
    data_loader: DataLoader,
    c_infos: Iterable[InteractionRotation],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Float[torch.Tensor, "batch ... rotated"]]:
    def get_module_name(m_name):
        try:
            get_model_attr(hooked_model.model, m_name)
            return m_name
        except AttributeError:
            return hooked_model.model.module_id_to_section_id[m_name]

    hooks = [
        Hook(
            name="rotated_acts",
            data_key=c_info.node_layer_name,
            fn=rotated_acts_pre_forward_hook_fn,
            module_name=get_module_name(c_info.node_layer_name),
            fn_kwargs={"rotation_matrix": c_info.C.to("cuda")},
        )
        for c_info in c_infos
        if c_info.C is not None
    ]

    with torch.inference_mode():
        run_dataset_through_model(
            hooked_model,
            data_loader,
            hooks,
            dtype=dtype,
            device=device,
            use_tqdm=True,
        )

    rib_acts = {
        m_name: torch.concatenate(act_list, dim=0)
        for m_name, act_list in hooked_model.hooked_data["rotated_acts"].items()
    }
    hooked_model.clear_hooked_data()
    return rib_acts
