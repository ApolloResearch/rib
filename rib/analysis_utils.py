from typing import Any, Iterable

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from rib.data_accumulator import run_dataset_through_model
from rib.hook_fns import rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.interaction_algos import InteractionRotation


def parse_c_infos(c_infos: list[dict[str, Any]]) -> dict[str, InteractionRotation]:
    """Converts the list of dicts from loading rib results into a dict of InteractionRotations."""
    return {c_info["node_layer_name"]: InteractionRotation(**c_info) for c_info in c_infos}


def get_rib_acts(
    hooked_model: HookedModel,
    data_loader: DataLoader,
    c_infos: Iterable[InteractionRotation],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Float[torch.Tensor, "batch ... rotated"]]:
    """Returns the activations in rib space when the model is run on the dataset.

    Will be (cpu) memory intensive if the dataset is large."""

    def get_module_name(m_name):
        if hasattr(hooked_model.model, "module_id_to_section_id"):
            return hooked_model.model.module_id_to_section_id[m_name]
        else:
            return m_name

    hooks = [
        Hook(
            name="rotated_acts",
            data_key=c_info.node_layer_name,
            fn=rotate_pre_forward_hook_fn,
            module_name=get_module_name(c_info.node_layer_name),
            fn_kwargs={"rotation_matrix": c_info.C.to(device), "mode": "cache"},
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
