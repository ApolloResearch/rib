"""Functions used to accumulate certain data when passing batches through a model."""


from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from rib.hook_manager import Hook, HookConfig, HookedModel


def run_dataset_through_model(
    hooked_model: HookedModel, dataloader: DataLoader, hooks: list[Hook], device: str = "cuda"
) -> None:
    """Simply pass all batches through a hooked model."""
    assert len(hooks) > 0, "Hooks have not been applied to this model."
    for batch in dataloader:
        data, _ = batch
        data = data.to(device)
        hooked_model(data, hooks=hooks)


def collect_gram_matrices(
    hooked_mlp: HookedModel,
    hook_configs: list[HookConfig],
    data_loader: DataLoader,
    device: str,
) -> dict[str, Float[Tensor, "d_hidden d_hidden"]]:
    """Collect gram matrices for each hook config.

    Args:
        hooked_mlp: The hooked model.
        hook_configs: The configs for the hook points.
        data_loader: The pytorch data loader.
        device: The device to run the model on.

    Returns:
        A dictionary of gram matrices, where the keys are the hook names.
    """
    assert len(hook_configs) > 0, "No hook configs provided."
    gram_hooks: list[Hook] = []
    for hook_config in hook_configs:
        assert hook_config.hook_type in ["forward", "pre_forward"]
        gram_hook_fn_name = f"gram_{hook_config.hook_type}_hook_fn"
        gram_hooks.append(
            Hook(
                name=hook_config.hook_name,
                data_key="gram",
                fn_name=gram_hook_fn_name,
                module_name=hook_config.module_name,
            )
        )

    run_dataset_through_model(hooked_mlp, data_loader, gram_hooks, device=device)

    gram_matrices: dict[str, Float[Tensor, "d_hidden d_hidden"]] = {
        hook_name: hooked_mlp.hooked_data[hook_name]["gram"] for hook_name in hooked_mlp.hooked_data
    }
    hooked_mlp.clear_hooked_data()

    len_dataset = len(data_loader.dataset)  # type: ignore

    # Scale the gram matrix by the number of samples in the dataset.
    for hook_name in gram_matrices:
        gram_matrices[hook_name] = gram_matrices[hook_name] / len_dataset

    return gram_matrices
