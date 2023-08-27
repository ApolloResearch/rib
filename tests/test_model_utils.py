from typing import TypeVar

import pytest

from rib.models.utils import create_list_partitions

T = TypeVar("T")


class TestCreatePartitions:
    @pytest.mark.parametrize(
        "in_list, sub_list, expected_output",
        [
            (
                ["embed", "pos_embed", "add_embed", "ln1.0", "attn.0", "add_resid1.0"],
                ["ln1.0", "add_resid1.0"],
                [["embed", "pos_embed", "add_embed"], ["ln1.0", "attn.0"], ["add_resid1.0"]],
            ),
            # Elements in sub_list are at the start and end of in_list
            (["a", "b", "c", "d"], ["a", "d"], [["a", "b", "c"], ["d"]]),
            # All elements in in_list are in sub_list
            ([0, 1, 2], [0, 1, 2], [[0], [1], [2]]),
        ],
    )
    def test_create_list_partitions(
        self, in_list: list[T], sub_list: list[T], expected_output: list[list[T]]
    ):
        result = create_list_partitions(in_list, sub_list)
        assert result == expected_output

    @pytest.mark.parametrize(
        "in_list, sub_list",
        [(["a", "b", "c"], ["a", "x"])],
    )
    def test_create_list_partitions_invalid(self, in_list: list[T], sub_list: list[T]):
        with pytest.raises(AssertionError):
            create_list_partitions(in_list, sub_list)
