import torch
from torch.testing import assert_close


def assert_is_close(a, b, atol, rtol, **kwargs):
    """Customized version of torch.testing.assert_close(). **kwargs added in msg output"""
    kwargs_str = "\n".join([f"{k}={v}" for k, v in kwargs.items()])
    msg_modifier = lambda m: m + "\n" + kwargs_str
    b = torch.as_tensor(b)
    b = torch.broadcast_to(b, a.shape)
    assert_close(
        a, b, atol=atol, rtol=rtol, msg=msg_modifier, check_device=False, check_dtype=False
    )


def assert_is_ones(tensor, atol, **kwargs):
    """Assert that all elements of a tensor are 1. **kwargs added in msg output"""
    assert_is_close(tensor, 1.0, atol=atol, rtol=0, **kwargs)


def assert_is_zeros(tensor, atol, **kwargs):
    """Assert that all elements of a tensor are 0. **kwargs added in msg output"""
    assert_is_close(tensor, 0.0, atol=atol, rtol=0, **kwargs)
