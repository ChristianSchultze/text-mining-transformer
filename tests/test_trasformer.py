"""Test module for the attention is all you need transformer and its modules."""
import torch
from src.transformer.model import SelfAttention, CrossAttention


class TestTransformer:
    """Test Transformer Modules"""

    def test_embedding(self):
        # TODO: implement
        pass

    def test_attention(self):
        test_model_dim = 512
        ground_truth = (test_model_dim, 17)
        test_num_heads = 8
        test_tensor = torch.zeros(ground_truth)
        test_attention = SelfAttention(num_heads=test_num_heads, model_dim=test_model_dim)
        result = test_attention.forward(test_tensor)
        assert result.shape == ground_truth

    def test_cross_attention(self):
        test_model_dim = 512
        ground_truth = (test_model_dim, 17)
        test_num_heads = 8
        test_tensor = torch.zeros(ground_truth)
        test_attention = CrossAttention(num_heads=test_num_heads, model_dim=test_model_dim)
        test_attention.encoder_output = torch.zeros(ground_truth)
        result = test_attention.forward(test_tensor)
        assert result.shape == ground_truth

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
        data = torch.zeros((heads * dim, tokens))
        # keys = torch.zeros((tokens, heads * dim))

        att = SelfAttention(heads, dim)
        data = att.transpose_for_multi_head(data)
        assert data.shape == (heads, dim, tokens)

        # keys = att.transpose_for_multi_head(keys)

        # torch.matmul(data, keys.transpose(-1, -2))
