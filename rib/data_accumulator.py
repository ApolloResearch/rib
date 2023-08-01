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

    # Scale the gram matrix by the number of samples in the dataset.
    len_dataset = len(data_loader.dataset)  # type: ignore
    for hook_name in gram_matrices:
        gram_matrices[hook_name] = gram_matrices[hook_name] / len_dataset

    return gram_matrices


def collect_M_dash_and_Lambda_dash(
    next_layer_C: Float[Tensor, "out_hidden out_hidden"],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    hook_config: HookConfig,
    device: str,
) -> tuple[Float[Tensor, "in_hidden in_hidden"], Float[Tensor, "in_hidden in_hidden"]]:
    """Collect the matrices M' and Lambda' for the layer specified in the hook config.

    We accumulate the matrices O (the jacobian), M' and Lambda' for each batch. To do this, we apply
    a hook to the module specified in the hook config. This hook will accumulate all 3 matrices
    over the batches.

    Args:
        next_layer_C: The rotation matrix for the next layer.
        hooked_model: The hooked model.
        data_loader: The data loader.
        hook_config: The hook config for the layer/module we are collecting the matrices for.
        device: The device to run the model on.

    Returns:
        A tuple containing M' and Lambda'.
    """
    assert hook_config.hook_type == "forward", "This function only works for forward hooks."
    interaction_hook_fn_name = f"M_dash_and_Lambda_dash_{hook_config.hook_type}_hook_fn"
    interaction_hook = Hook(
        name=hook_config.hook_name,
        data_key=["M_dash", "Lambda_dash"],
        fn_name=interaction_hook_fn_name,
        module_name=hook_config.module_name,
        fn_kwargs={
            "next_layer_C": next_layer_C,
        },
    )

    run_dataset_through_model(hooked_model, data_loader, hooks=[interaction_hook], device=device)

    M_dash = hooked_model.hooked_data[hook_config.hook_name]["M_dash"]
    Lambda_dash = hooked_model.hooked_data[hook_config.hook_name]["Lambda_dash"]
    hooked_model.clear_hooked_data()

    # Scale the matrices by the number of samples in the dataset.
    len_dataset = len(data_loader.dataset)  # type: ignore
    M_dash = M_dash / len_dataset
    Lambda_dash = Lambda_dash / len_dataset

    return M_dash, Lambda_dash
