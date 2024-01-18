from typing import Iterable

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from rib.data_accumulator import run_dataset_through_model
from rib.hook_fns import rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.interaction_algos import InteractionRotation


def rotation_list_to_dict(rotations: list[InteractionRotation]) -> dict[str, InteractionRotation]:
    """Converts a list of InteractionRotation objects to a dict keyed by node_layer_name."""
    return {info.node_layer_name: info for info in rotations}


def get_rib_acts(
    hooked_model: HookedModel,
    data_loader: DataLoader,
    interaction_rotations: Iterable[InteractionRotation],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Float[torch.Tensor, "batch ... rotated"]]:
    """Returns the activations in rib space when the model is run on the dataset.

    Will be (cpu) memory intensive if the dataset is large.
    """

    def get_module_name(m_name):
        if hasattr(hooked_model.model, "module_id_to_section_id"):
            return hooked_model.model.module_id_to_section_id[m_name]
        else:
            return m_name

    hooks = [
        Hook(
            name="rotated_acts",
            data_key=info.node_layer_name,
            fn=rotate_pre_forward_hook_fn,
            module_name=get_module_name(info.node_layer_name),
            fn_kwargs={"rotation_matrix": info.C.to(device=device, dtype=dtype), "mode": "cache"},
        )
        for info in interaction_rotations
        if info.C is not None
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
        m_name: torch.concatenate(act_list, dim=0).cpu()
        for m_name, act_list in hooked_model.hooked_data["rotated_acts"].items()
    }
    hooked_model.clear_hooked_data()
    return rib_acts
