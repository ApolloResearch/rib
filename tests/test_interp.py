from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from experiments.interp_modular_arithmetic.activations import Activations


def test_logits_correctness():
    """Test modular addition math is correct"""
    path = (
        Path(__file__).parent.parent
        / "experiments/interp_modular_arithmetic/mod_arithmetic_config.yaml"
    )
    activations = Activations(path)

    logits = activations.hooked_model(torch.tensor([[100, 42, 113]]))[0]
    answer = logits[0, -1].argmax().item()
    assert answer == (100 + 42) % 113, "Logits do not match the correct answer"


@pytest.mark.slow
def test_activations_batched_vs_unbatched():
    """Test that batched and unbatched activation collectors produce identical results"""
    path = (
        Path(__file__).parent.parent
        / "experiments/interp_modular_arithmetic/mod_arithmetic_config.yaml"
    )
    activations = Activations(path)

    resid_acts_sec_0_2 = activations.get_section_activations(
        section="sections.section_0.2",
    )[0]
    resid_acts_sec_0_2_nobatch = activations.get_section_activations_unbatched(
        section="sections.section_0.2"
    )[0]
    assert torch.allclose(
        resid_acts_sec_0_2, resid_acts_sec_0_2_nobatch, atol=1e-6
    ), "Batched and unbatched activations do not match"


def test_attention_pattern():
    """Test that attention pattern obeys causal mask"""
    path = (
        Path(__file__).parent.parent
        / "experiments/interp_modular_arithmetic/mod_arithmetic_config.yaml"
    )
    activations = Activations(path)

    attention_scores = activations.get_section_activations(
        section="sections.section_0.1.attention_scores",
    )[0]
    attention_scores_x_to_y = attention_scores[:, :, :, 0, 1]
    assert torch.allclose(
        attention_scores_x_to_y, torch.tensor(-1e5)
    ), "Acausal attention scores from x to y are not 1e-5"
