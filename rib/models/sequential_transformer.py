"""
Defines a Transformer based on the transformer lens but with a flattened, sequential structure.
"""
from typing import Any, Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from rib.models.utils import ACTIVATION_MAP

"""

Embed
Positional Encoding
LayerNormPreFolded
Attention
ADD_RESIDUAL
LayerNormPreFolded
MLP_IN [linear]
MLP_ACT [relu]
MLP_OUT [linear]
ADD_RESIDUAL [custom]
UnEmbed

"""


class Transformer:
    def __init__(self, config):
        module_list = self.build_modules(config)
        self.modules = torch.nn.Sequential(**module_list)

    def build_modules(self, config):
        """Build the list of modules using the config file."""
        module_list = []
        return module_list

    def forward(self, x):
        self.modules(x)


class Embed:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class PosEmbed:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class Unembed:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class AddResidual:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class LayerNormPreFolded:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class Attention:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class MlpIN:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class MlpAct:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x


class MlpOut:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return x
