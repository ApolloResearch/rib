from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.hook_fns import rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix
from rib.utils import eval_model_accuracy


def ablate_and_test(
    hooked_model: HookedModel,
    module_name: str,
    interaction_rotation: Float[Tensor, "d_hidden d_hidden_trunc"],
    interaction_rotation_pinv: Float[Tensor, "d_hidden_trunc d_hidden"],
    test_loader: DataLoader,
    device: str,
    ablation_schedule: list[int],
    dtype: torch.dtype = torch.float32,
    hook_name: Optional[str] = None,
) -> dict[int, float]:
    """Ablate eigenvectors and test the model accuracy.

    Args:
        hooked_model: The hooked model.
        module_name: The name of the module whose inputs we want to rotate and ablate.
        interaction_rotation: The matrix that rotates activations to the interaction basis. (C)
        interaction_rotation_pinv: The pseudo-inverse of the interaction rotation matrix. (C^+)
        hook_config: The config for the hook point.
        test_loader: The DataLoader for the test data.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        ablation_schedule: A list of the number of vectors to ablate at each step.
        hook_name: The name of the hook point to use. If None, defaults to `module_name`.

    Returns:
        Dictionary mapping the number of ablated vectors to the resulting accuracy.
    """

    if hook_name is None:
        hook_name = module_name

    accuracies: dict[int, float] = {}
    # Iterate through possible number of ablated vectors.
    for n_ablated_vecs in tqdm(
        ablation_schedule,
        total=len(ablation_schedule),
        desc=f"Ablating {module_name}",
    ):
        interaction_rotation = interaction_rotation.to(device)
        interaction_rotation_pinv = interaction_rotation_pinv.to(device)
        rotation_matrix = calc_rotation_matrix(
            vecs=interaction_rotation,
            vecs_pinv=interaction_rotation_pinv,
            n_ablated_vecs=n_ablated_vecs,
        )
        rotation_hook = Hook(
            name=hook_name,
            data_key="rotation",
            fn=rotate_pre_forward_hook_fn,
            module_name=module_name,
            fn_kwargs={"rotation_matrix": rotation_matrix},
        )

        accuracy_ablated = eval_model_accuracy(
            hooked_model, test_loader, hooks=[rotation_hook], dtype=dtype, device=device
        )
        accuracies[n_ablated_vecs] = accuracy_ablated

    return accuracies
