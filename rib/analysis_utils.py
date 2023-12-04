from typing import Any, Iterable

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from rib.data_accumulator import run_dataset_through_model
from rib.hook_fns import rotated_acts_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.interaction_algos import InteractionRotation


def parse_c_infos(c_infos: list[dict[str, Any]]) -> dict[str, InteractionRotation]:
    return {c_info["node_layer_name"]: InteractionRotation(**c_info) for c_info in c_infos}


def get_rib_acts(
    hooked_model: HookedModel, data_loader: DataLoader, c_infos: Iterable[InteractionRotation]
) -> dict[str, Float[torch.Tensor, ["batch", "..."]]]:
    hooks = [
        Hook(
            name="rotated_acts",
            data_key=c_info.module_id,
            fn=rotated_acts_forward_hook_fn,
            module_name=hooked_model.model.module_id_to_section_id[c_info.module_id],
            fn_kwargs={"rotation_matrix": c_info.C.to("cuda"), "output_rotated": False},
        )
        for c_info in c_infos
    ]
    with torch.inference_mode():
        run_dataset_through_model(
            hooked_model, data_loader, hooks, dtype=torch.float64, device="cuda", use_tqdm=True
        )

    return {
        c_info.module_id: torch.concatenate(
            hooked_model.hooked_data["rotated_acts"][c_infos.module_id], dim=0
        )
        for c_info in c_infos
    }
