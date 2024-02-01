import pytest

from rib.data import (
    BlockVectorDatasetConfig,
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.utils import replace_pydantic_model


def test_hf_dataset_config_validation():
    """Test the validation of the HFDatasetConfig model.

    For HF datasets, we can't have both return_set_frac and n_documents be non-None, but we can
    have all other combinations.
    """
    base_config = HFDatasetConfig(
        dataset_type="huggingface",
        name="test",
        tokenizer_name="test",
        return_set="train",
    )
    valid_combinations = [
        {"return_set_frac": 0.5, "n_samples": None, "n_documents": None},
        {"return_set_frac": None, "n_samples": 10, "n_documents": None},
        {"return_set_frac": None, "n_samples": None, "n_documents": 10},
        {"return_set_frac": 0.5, "n_samples": None, "n_documents": 10},
    ]
    for combination in valid_combinations:
        replace_pydantic_model(base_config, combination)

    with pytest.raises(ValueError):
        # invalid combination
        replace_pydantic_model(
            base_config, {"return_set_frac": 0.5, "n_samples": 10, "n_documents": 10}
        )


def test_non_hf_dataset_config_validation():
    """Test the validation of dataset configs that are not HFDatasetConfig.

    We can't have both return_set_frac and n_samples be non-None.
    """
    for base_config in [
        BlockVectorDatasetConfig(dataset_type="block_vector"),
        VisionDatasetConfig(dataset_type="torchvision"),
        ModularArithmeticDatasetConfig(dataset_type="modular_arithmetic"),
    ]:
        valid_combinations = [
            {"return_set_frac": 0.5, "n_samples": None},
            {"return_set_frac": None, "n_samples": 10},
            {"return_set_frac": None, "n_samples": None},
        ]
        for combination in valid_combinations:
            replace_pydantic_model(base_config, combination)

        with pytest.raises(ValueError):
            # invalid combination
            replace_pydantic_model(base_config, {"return_set_frac": 0.5, "n_samples": 10})
