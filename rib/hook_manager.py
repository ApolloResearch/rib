"""Defines a Hook object and a HookedModel class for adding hooks to PyTorch models.

A Hook object defines a hook function and the hook point to add the hook to. The HookedModel class
is a wrapper around a PyTorch model that allows hooks to be added and removed.
"""

import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional, Union

import torch

from rib.hook_fns import acts_forward_hook_fn
from rib.models.mlp import MLP
from rib.models.utils import get_model_attr


@dataclass
class Hook:
    """Defines a hook object that can be added to a model.

    After initialization, the hook_type is created and stored as an attribute based on whether
    fn contains an output argument.


    Attributes:
        name: Name of the hook. This is useful for identifying hooks when two hooks have the
            same module_name (e.g. a forward and pre_forward hook).
        data_key: The key or keys to index the data in HookedModel.hooked_data.
        fn: The hook function to run at the hook point.
        module_name: String representing the attribute of the model to add the hook to.
            Nested attributes are split by periods (e.g. "layers.linear_0").
        fn_kwargs: Additional keyword arguments to pass to the hook function.
    """

    name: str
    data_key: Union[str, list[str]]
    fn: Callable
    module_name: str
    fn_kwargs: dict[str, Any] = field(default_factory=dict)
    hook_type: str = "forward"

    def __post_init__(self) -> None:
        """Set the hook_type attribute based on whether fn contains an output argument.

        Also verify that the name of the function contains one of
        'forward','pre_forward','backward','pre_backward' depending on which type is inferred.
        """
        fn_args = list(inspect.signature(self.fn).parameters.keys())
        forward_args = ["module", "inputs", "output"]
        pre_forward_args = ["module", "inputs"]
        backward_args = ["module", "grad_input", "grad_output"]
        pre_backward_args = ["module", "grad_output"]
        if len(fn_args) > 2 and fn_args[:3] == forward_args:
            self.hook_type = "forward"
            assert (
                "forward" in self.fn.__name__ and "pre_forward" not in self.fn.__name__
            ), f"Hook name must contain 'forward' for forward hooks, got {self.fn.__name__}"
        elif len(fn_args) > 1 and fn_args[:2] == pre_forward_args:
            self.hook_type = "pre_forward"
            assert (
                "pre_forward" in self.fn.__name__
            ), f"Hook name must contain 'pre_forward' for pre_forward hooks, got {self.fn.__name__}"
        elif len(fn_args) > 2 and fn_args[:3] == backward_args:
            self.hook_type = "backward"
            assert (
                "backward" in self.fn.__name__ and "pre_backward" not in self.fn.__name__
            ), f"Hook name must contain 'backward' for backward hooks, got {self.fn.__name__}"
        elif len(fn_args) > 1 and fn_args[:2] == pre_backward_args:
            self.hook_type = "pre_backward"
            assert (
                "pre_backward" in self.fn.__name__
            ), f"Hook name must contain 'pre_backward' for pre_backward hooks, got {self.fn.__name__}"
        else:
            raise ValueError(
                f"Hook function must have signature (module, inputs, [output]) or "
                f"(module, grad_input, grad_output) or (module, grad_output), got {fn_args}"
            )

        # if self.hook_type == "forward":
        #     assert fn_args[:2] == [
        #         "module",
        #         "inputs",
        #     ], f"Hook function must have signature (module, inputs, ...), got {fn_args}"
        #     if len(fn_args) > 2 and fn_args[2] == "output":
        #         self.hook_type = "forward"
        #         assert (
        #             "forward" in self.fn.__name__ and "pre_forward" not in self.fn.__name__
        #         ), f"Hook name must contain 'forward' for forward hooks, got {self.fn.__name__}"
        #     else:
        #         self.hook_type = "pre_forward"
        #         assert "pre_forward" in self.fn.__name__, (
        #             f"Hook name must contain 'pre_forward' for pre_forward hooks, got "
        #             f"{self.fn.__name__}"
        #         )
        # elif self.hook_type == "backward":
        #     assert fn_args[:2] == [
        #         "module",
        #         "inputs",
        #     ], f"Hook function must have signature (module, inputs, ...), got {fn_args}"
        #     if len(fn_args) > 2 and fn_args[1] == "grad_input":
        #         self.hook_type = "backward"
        #         assert "backward" in self.fn.__name__, (
        #             f"Hook name must contain 'backward' for pre_backward hooks, got "
        #             f"{self.fn.__name__}"
        #         )
        #     else:
        #         self.hook_type = "pre_backward"
        #         assert (
        #             "pre_backward" in self.fn.__name__
        #         ), f"Hook name must contain 'pre_backward' for backward hooks, got {self.fn.__name__}"
        # else:
        #     raise ValueError(f"hook_type must be 'forward' or 'backward', got {self.hook_type}")


class HookedModel(torch.nn.Module):
    """A wrapper around a PyTorch model that allows hooks to be added and removed.

    Example:
        >>> model = torch.nn.Sequential()
        >>> model.add_module("linear_0", torch.nn.Linear(3, 2))
        >>> hooked_model = HookedModel(model)
        >>> hook = Hook(name="forward_linear_0", data_key="gram", fn=gram_forward_hook_fn,
            module_name="linear_0")
        >>> hooked_model(torch.randn(6, 3), hooks=[hook])
        >>> hooked_model.hooked_data["linear_0"]["gram"]
        tensor([[ 1.2023, -0.0311],
                [-0.0311,  0.9988]])
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.forward_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.backward_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.hooked_data: dict[str, Any] = {}

    def __call__(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        return self.forward(*args, forward_hooks=hooks, **kwargs)

    def forward(self, *args, forward_hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        """Run the forward pass of the model and remove all hooks."""
        if forward_hooks is not None:
            self.add_hooks(forward_hooks)
        try:
            output = self.model(*args, **kwargs)
        finally:
            self.remove_forward_hooks()
        return output

    def add_hooks(self, hooks: list[Hook]) -> None:
        """Add a hook to the model at each of the specified hook points."""
        for hook in hooks:
            hook_module = get_model_attr(self.model, hook.module_name)
            hook_fn_partial: partial = partial(
                hook.fn,
                hooked_data=self.hooked_data,
                hook_name=hook.name,
                data_key=hook.data_key,
                **hook.fn_kwargs,
            )
            if hook.hook_type == "forward":
                handle = hook_module.register_forward_hook(hook_fn_partial)
                self.forward_hook_handles.append(handle)
            elif hook.hook_type == "pre_forward":
                handle = hook_module.register_forward_pre_hook(hook_fn_partial)
                self.forward_hook_handles.append(handle)
            elif hook.hook_type == "backward":
                handle = hook_module.register_full_backward_hook(hook_fn_partial)
                self.backward_hook_handles.append(handle)
            elif hook.hook_type == "pre_backward":
                handle = hook_module.register_full_backward_pre_hook(hook_fn_partial)
                self.backward_hook_handles.append(handle)
            else:
                raise ValueError(f"Invalid hook_type, got {hook.hook_type}")

    def remove_forward_hooks(self) -> None:
        """Remove all forward hooks from the model."""
        for handle in self.forward_hook_handles:
            handle.remove()
        self.forward_hook_handles = []

    def remove_backward_hooks(self) -> None:
        """Remove all backward hooks from the model."""
        for handle in self.backward_hook_handles:
            handle.remove()
        self.backward_hook_handles = []

    def clear_hooked_data(self) -> None:
        """Clear all data stored in the hooked_data attribute."""
        self.hooked_data = {}

    def run_with_cache(self, *args, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Run the forward pass of the model and return the output and all internal activations."""
        # We don't care about the activations of the root modules, and not e.g. modules that contain
        # other modules (e.g. Sequential or ModuleList)
        has_children: Callable[[torch.nn.Module], bool] = (
            lambda module: sum(1 for _ in module.children()) > 0
        )
        if isinstance(self.model, MLP):
            module_names = [f"layers.{i}" for i in range(len(self.model.layers))]
        else:
            # We use this check rather than has_children because we still want
            # AttentionOut even though it has the child AttentionScores.
            module_names = [
                name for name, mod in self.model.named_modules() if name.count(".") >= 2
            ]

        act_hooks: list[Hook] = []
        for module_name in module_names:
            act_hooks.append(
                Hook(
                    name=module_name,
                    data_key="acts",
                    fn=acts_forward_hook_fn,
                    module_name=module_name,
                )
            )
        output = self.forward(*args, forward_hooks=act_hooks, **kwargs)
        hooked_data = self.hooked_data
        self.clear_hooked_data()
        return output, hooked_data
