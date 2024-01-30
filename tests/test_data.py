import pytest

from rib.data import HFDatasetConfig
from rib.utils import replace_pydantic_model


def test_invalid_hf_dataset_config():
    """Test that invalid combinations of attributes in HFDatasetConfig are caught."""
    base_config = HFDatasetConfig(
        dataset_type="huggingface",
        name="test",
        tokenizer_name="test",
        return_set="train",
    )

    with pytest.raises(ValueError):
        # return_set_frac and return_set_n_samples cannot be used together
        replace_pydantic_model(base_config, {"return_set_frac": 0.5, "return_set_n_samples": 10})
        # return_set_n_documents and return_set_frac cannot be used
        replace_pydantic_model(base_config, {"return_set_frac": 0.5, "return_set_n_documents": 10})
        # If return_set_n_documents is used, return_set_n_samples must be not None
        replace_pydantic_model(base_config, {"return_set_n_documents": 10})
        # If return_set_n_documents is used, return_set_frac must be None
        replace_pydantic_model(base_config, {"return_set_n_documents": 10, "return_set_frac": 0.5})
