import pytest
import torch

from rib.hook_fns import _concatenate_with_embedding_reshape


def test_concatenate_with_embedding_reshape():
    # Define some sample tensors
    tensor_2d = torch.randn(4, 8)
    tensor_3d_1 = torch.randn(4, 5, 8)
    tensor_3d_2 = torch.randn(4, 3, 8)

    # 1. Test a 2D tensor (no positional embeddings)
    result = _concatenate_with_embedding_reshape((tensor_2d,))
    assert result.size() == (4, 8)
    assert torch.equal(result, tensor_2d)

    # 2. Test a 3D tensor (with positional embeddings)
    result = _concatenate_with_embedding_reshape((tensor_3d_1,))
    expected_shape = (4, 5 * 8)
    assert result.size() == expected_shape

    # 3. Test two 3D tensors
    result = _concatenate_with_embedding_reshape((tensor_3d_1, tensor_3d_2))
    expected_shape = (4, 5 * 8 + 3 * 8)
    assert result.size() == expected_shape

    # 4. Test tensor with unsupported rank (e.g., 1D tensor)
    tensor_1d = torch.randn(8)
    with pytest.raises(ValueError, match="Unexpected tensor rank"):
        _concatenate_with_embedding_reshape((tensor_1d,))

    # 5. Test tensor combination with unsupported rank
    with pytest.raises(ValueError, match="Unexpected tensor rank"):
        _concatenate_with_embedding_reshape((tensor_2d, tensor_1d))
