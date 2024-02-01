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

    # valid combinations
    replace_pydantic_model(
        base_config, {"return_set_frac": 0.5, "n_samples": 10, "n_documents": None}
    )
    replace_pydantic_model(
        base_config, {"return_set_frac": None, "n_samples": 10, "n_documents": 10}
    )
    replace_pydantic_model(
        base_config, {"return_set_frac": None, "n_samples": None, "n_documents": 10}
    )
    replace_pydantic_model(
        base_config, {"return_set_frac": 0.5, "n_samples": None, "n_documents": None}
    )

    # invalid combination
    with pytest.raises(ValueError):
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
        # valid combinations
        replace_pydantic_model(base_config, {"return_set_frac": 0.5, "n_samples": None})
        replace_pydantic_model(base_config, {"return_set_frac": None, "n_samples": 10})
        replace_pydantic_model(base_config, {"return_set_frac": None, "n_samples": None})

        # invalid combination
        with pytest.raises(ValueError):
            replace_pydantic_model(base_config, {"return_set_frac": 0.5, "n_samples": 10})
