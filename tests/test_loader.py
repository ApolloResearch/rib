import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from rib.loader import load_sequential_transformer, prepare_dataset
from rib.utils import get_data_subset, set_seed


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


@pytest.mark.slow()
@pytest.mark.parametrize("model_str", ["tiny-stories-1M", "gpt2", "pythia-14M"])
def test_load_transformer(model_str):
    """Test load_sequential_transformer runs without error.

    We test for equality with the tlens model in `test_tlens_converter` but it doesn't run
    the load_sequential_transformer function."""
    _ = load_sequential_transformer(
        node_layers=["ln2.1", "unembed"],
        tlens_pretrained=model_str,
        last_pos_module_type=None,
        tlens_model_path=None,
        device="cpu",
        fold_bias=True,
    )


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0  # Example EOS token ID
        self.generator = torch.Generator().manual_seed(0)

    def __call__(self, text: str) -> dict[str, list[int]]:
        # Generate between 5 to 10 token IDs as an example
        num_tokens = torch.randint(5, 10, (1,), generator=self.generator).item()
        token_ids = torch.randint(1, 100, (num_tokens,), generator=self.generator).tolist()
        return {"input_ids": token_ids}


class TestTokenizeDataset:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.sample_texts = ["This is a test.", "Another test sentence."]
        self.sample_dataset = [{"text": text} for text in self.sample_texts]
        # Create a dummy tokenizer that spits out random tokens
        set_seed(0)
        self.tokenizer = MockTokenizer()

    def test_outputs_are_all_n_ctx_length(self):
        n_ctx = 5
        tokenized_dataset = prepare_dataset(self.sample_dataset, self.tokenizer, n_ctx)
        for input_ids, labels in tokenized_dataset:
            assert len(input_ids) == n_ctx
            assert len(labels) == n_ctx

    def test_dataset_has_expected_size(self):
        n_ctx = 5
        n_samples = 3
        tokenized_dataset = prepare_dataset(self.sample_dataset, self.tokenizer, n_ctx, n_samples)
        assert len(tokenized_dataset) == n_samples

    def test_seed_reproducibility(self):
        n_ctx = 5
        n_samples = 2
        seed = 0
        dataset1 = prepare_dataset(self.sample_dataset, self.tokenizer, n_ctx, n_samples, seed)
        duplicate_tokenizer = MockTokenizer()
        dataset2 = prepare_dataset(self.sample_dataset, duplicate_tokenizer, n_ctx, n_samples, seed)
        assert torch.equal(dataset1.tensors[0], dataset2.tensors[0]) and torch.equal(
            dataset1.tensors[1], dataset2.tensors[1]
        )

    def test_different_seeds(self):
        n_ctx = 5
        n_samples = 2
        dataset1 = prepare_dataset(self.sample_dataset, self.tokenizer, n_ctx, n_samples, 42)
        duplicate_tokenizer = MockTokenizer()
        dataset2 = prepare_dataset(self.sample_dataset, duplicate_tokenizer, n_ctx, n_samples, 43)
        assert not torch.equal(dataset1.tensors[0], dataset2.tensors[0]) or not torch.equal(
            dataset1.tensors[1], dataset2.tensors[1]
        )

    def test_input_ids_equal_labels_no_sampling(self):
        """If not sampling (i.e. n_samples is None), input_ids and labels differ by one token.

        Moreover, the final label of one chunk is the input_id of the first token in the next chunk.
        So we can flatten the input_ids and labels and check that they are equal (offset by one).
        """
        n_ctx = 5
        tokenized_dataset = prepare_dataset(self.sample_dataset, self.tokenizer, n_ctx)
        flattened_input_ids = [
            token_id for input_ids, _ in tokenized_dataset for token_id in input_ids
        ]
        flattened_labels = [token_id for _, labels in tokenized_dataset for token_id in labels]
        assert len(flattened_input_ids) == len(flattened_labels)
        assert flattened_input_ids[1:] == flattened_labels[:-1]

    def test_input_ids_equal_labels_sampling(self):
        """Check that the labels match the input_ids except for the final token when sampling.

        When (randomly) sampling, the chunks will not be ordered, so we can't check that the final
        token label is the input_id of the first token in the next chunk.
        """
        n_ctx = 5
        n_samples = 3
        tokenized_dataset = prepare_dataset(self.sample_dataset, self.tokenizer, n_ctx, n_samples)
        for input_ids, labels in tokenized_dataset:
            assert len(input_ids) == len(labels)
            assert torch.equal(input_ids[1:], labels[:-1])
