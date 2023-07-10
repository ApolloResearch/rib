from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import torch

from rib.hook_registry import HOOK_REGISTRY
from rib.models.utils import get_model_attr


@dataclass
class Hook:
    """Defines a hook object that can be added to a model.

    After initialization, the hook function and type are stored in the fn and hook_type attributes,
    respectively.


    Attributes:
        name: Name of the hook. This is useful for identifying hooks when two hooks have the
            same module_name (e.g. a forward and pre_forward hook).
        data_key: The key used to store data in HookedModel.hookd_data.
        fn_name: Name of the hook function to run at the hook point.
        module_name: String representing the attribute of the model to add the hook to.
            Nested attributes are split by periods (e.g. "layers.linear_0").
        fn_kwargs: Additional keyword arguments to pass to the hook function.
    """

    name: str
    data_key: str
    fn_name: str
    module_name: str
    fn_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_fn_name()
        self.fn, self.hook_type = HOOK_REGISTRY[self.fn_name]

    def validate_fn_name(self):
        if self.fn_name not in HOOK_REGISTRY:
            raise ValueError(
                f"fn_name must be one of {list(HOOK_REGISTRY.keys())}, got {self.fn_name}"
            )


class HookedModel(torch.nn.Module):
    """A wrapper around a PyTorch model that allows hooks to be added and removed.

    Example:
        >>> model = torch.nn.Sequential()
        >>> model.add_module("linear_0", torch.nn.Linear(3, 2))
        >>> hooked_model = HookedModel(model)
        >>> hook = Hook(name="forward_linear_0", data_key="gram", fn_name="gram_forward_hook_fn",
            module_name="linear_0")
        >>> hooked_model(torch.randn(6, 3), hooks=[hook])
        >>> hooked_model.hooked_data["linear_0"]["gram"]
        tensor([[ 1.2023, -0.0311],
                [-0.0311,  0.9988]])
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.hooked_data: dict[str, Any] = {}

    def __call__(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        return self.forward(*args, hooks=hooks, **kwargs)

    def forward(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        """Run the forward pass of the model and remove all hooks."""
        if hooks is not None:
            self.add_hooks(hooks)
        try:
            output = self.model(*args, **kwargs)
        finally:
            self.remove_hooks()
        return output

    def add_hooks(self, hooks: list[Hook]) -> None:
        """Add a hook to the model at each of the specified hook points."""
        for hook in hooks:
            hook_module = get_model_attr(self.model, hook.module_name)
            hook_fn_partial = partial(
                hook.fn,
                hooked_data=self.hooked_data,
                hook_name=hook.name,
                data_key=hook.data_key,
                **hook.fn_kwargs,
            )
            if hook.hook_type == "forward":
                handle = hook_module.register_forward_hook(hook_fn_partial)
            elif hook.hook_type == "pre_forward":
                handle = hook_module.register_forward_pre_hook(hook_fn_partial)
            self.hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all hooks from the model."""
        for handle in self.hook_handles:
            handle.remove()
