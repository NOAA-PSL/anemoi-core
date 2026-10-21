# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Kept apart from test_attention.py, which needs hypothesis (not in the anemoi-rocm env, where flex
# attention is the backend to use).

import pickle

import pytest
import torch

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.utils import load_layer_kernels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# flex attention is compiled; inductor's CPU lowering of it fails (torch 2.10: IndexError), and the
# backend is meant for GPUs without flash attention anyway
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled flex attention needs a GPU")


@pytest.fixture(scope="module")
def layer_kernels():
    return load_layer_kernels()


def _pair(layer_kernels, window_size, num_heads=4, embed_dim=64):
    """Flex and SDPA self-attention with the same weights."""
    flex = MultiHeadSelfAttention(
        num_heads, embed_dim, layer_kernels, attention_implementation="flex_attention", window_size=window_size
    ).to(DEVICE)
    sdpa = MultiHeadSelfAttention(
        num_heads,
        embed_dim,
        layer_kernels,
        attention_implementation="scaled_dot_product_attention",
        window_size=window_size,
    ).to(DEVICE)
    sdpa.load_state_dict(flex.state_dict())
    return flex, sdpa


# grid 300 is not a multiple of flex attention's 128 block size, so the last block is partial
@needs_gpu
@pytest.mark.parametrize("window_size", [16, 100, None])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_flex_matches_sdpa(layer_kernels, window_size, batch_size):
    torch.manual_seed(0)
    grid, embed_dim = 300, 64
    flex, sdpa = _pair(layer_kernels, window_size, embed_dim=embed_dim)
    shard_info = GraphShardInfo(nodes=[grid])

    x = torch.randn(batch_size * grid, embed_dim, device=DEVICE, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = flex.forward(x, shard_info, batch_size)
    out_ref = sdpa.forward(x_ref, shard_info, batch_size)
    assert out.shape == x.shape
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    for (name, p), p_ref in zip(flex.named_parameters(), sdpa.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad, atol=1e-4, rtol=1e-4, msg=name)


@needs_gpu
def test_flex_window_changes_result(layer_kernels):
    torch.manual_seed(0)
    grid, embed_dim = 300, 64
    windowed, _ = _pair(layer_kernels, 16, embed_dim=embed_dim)
    full, _ = _pair(layer_kernels, None, embed_dim=embed_dim)
    full.load_state_dict(windowed.state_dict())
    x = torch.randn(grid, embed_dim, device=DEVICE)
    shard_info = GraphShardInfo(nodes=[grid])
    assert not torch.allclose(windowed.forward(x, shard_info, 1), full.forward(x, shard_info, 1), atol=1e-5)


@needs_gpu
def test_flex_pickles_after_forward(layer_kernels):
    """The model is pickled into checkpoints; the block mask and compiled function must stay out of it."""
    grid, embed_dim = 300, 64
    flex, _ = _pair(layer_kernels, 16, embed_dim=embed_dim)
    x = torch.randn(grid, embed_dim, device=DEVICE)
    shard_info = GraphShardInfo(nodes=[grid])
    out = flex.forward(x, shard_info, 1)
    assert flex.attention.block_mask is not None

    restored = pickle.loads(pickle.dumps(flex))
    assert restored.attention.block_mask is None
    assert flex.attention.block_mask is not None  # pickling does not touch the live module
    torch.testing.assert_close(restored.forward(x, shard_info, 1), out)


@pytest.mark.parametrize("kwargs", [{"dropout_p": 0.1}, {"softcap": 0.5}])
def test_flex_rejects_unsupported(layer_kernels, kwargs):
    flex = MultiHeadSelfAttention(
        4, 64, layer_kernels, attention_implementation="flex_attention", window_size=16, **kwargs
    ).to(DEVICE)
    flex.train()  # dropout is applied in training only
    x = torch.randn(300, 64, device=DEVICE)
    with pytest.raises(NotImplementedError):
        flex.forward(x, GraphShardInfo(nodes=[300]), 1)
