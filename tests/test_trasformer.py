"""Test module for the attention is all you need transformer and its modules."""
import torch
from src.transformer.model import SelfAttention


class TestTransformer:
    """Test Transformer Modules"""

    def test_embedding(self):
        # TODO: implement
        pass

    def test_attention(self):
        # TODO: implement
        pass

    def test_positional_encoding(self):
        # TODO: implement
        pass

    def test_feed_forward(self):
        # TODO: implement
        pass

    def test_transpose(self):
        dim = 8
        tokens = 4
        heads = 2
        data = torch.zeros((tokens, heads * dim))
        #keys = torch.zeros((tokens, heads * dim))

        att = SelfAttention(heads, dim)
        data = transpose_for_multi_head(data)
        assert data.shape == (heads, tokens, dim)

        #keys = att.transpose_for_multi_head(keys)

        #torch.matmul(data, keys.transpose(-1, -2))
