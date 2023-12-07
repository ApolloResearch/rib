import warnings

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from rib.utils import get_data_subset


@pytest.mark.parametrize(
    "dataset_size, frac, n_samples, seed, expected_length, expected_exception",
    [
        (10, 0.5, None, 42, 5, None),
        (10, None, 5, 42, 5, None),
        (10, None, None, 42, 10, None),
        (10, 0.5, 5, 42, None, AssertionError),
        (10, None, 15, 42, None, AssertionError),
        (10, 0.5, None, None, 5, None),  # No seed, results will be random
    ],
)
def test_get_data_subset(dataset_size, frac, n_samples, seed, expected_length, expected_exception):
    # Create a dummy dataset
    dataset = TensorDataset(torch.arange(start=0, end=dataset_size))

    if expected_exception:
        with pytest.raises(expected_exception):
            get_data_subset(dataset, frac, n_samples, seed)
    else:
        subset = get_data_subset(dataset, frac, n_samples, seed)
        expected_type = Subset if frac is not None or n_samples is not None else TensorDataset
        assert isinstance(subset, expected_type), "Returned object type is incorrect"

        if seed is not None or (frac is None and n_samples is None):
            # When seed is provided or neither frac nor n_samples is provided, expect a specific subset
            assert len(subset) == expected_length, "Length of subset is incorrect"
            if frac is not None or n_samples is not None:
                # Test reproducibility with seed
                subset_again = get_data_subset(dataset, frac, n_samples, seed)
                assert (
                    subset.indices == subset_again.indices
                ), "Subsets are not reproducible with the same seed"
        else:
            # When no seed is provided and either frac or n_samples is provided, expect a random subset
            assert len(subset) == expected_length, "Length of subset is incorrect"
