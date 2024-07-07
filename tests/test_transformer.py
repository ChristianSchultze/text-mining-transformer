"""Test module for the attention is all you need transformer and its modules."""
import torch
from src.transformer.model import SelfAttention, CrossAttention, FeedForward, PositionalEncoding, Embedding, Transformer


class TestTransformer:
    """Test Transformer Modules"""

    def test_transformer(self):
        model_dim = 128
        word_token_dim = 100
        num_layers = 6
        ff_dim = 256
        num_heads = 8
        gt_shape = (1, 100, 8)

        input_data = torch.ones(gt_shape)
        output_input = torch.ones(gt_shape)
        transformer = Transformer(word_token_dim, num_layers, model_dim, num_heads, ff_dim)

        result = transformer(input_data, output_input)

        assert result.shape == gt_shape


    def test_embedding(self):
        model_dim = 512
        word_token_dim = 2048
        token_count = 42

        ground_truth = (1, model_dim, 42)
        data = torch.zeros((1, word_token_dim, token_count))

        embedding = Embedding(word_token_dim, model_dim)
        result = embedding(data)

        assert result.shape == ground_truth

    def test_attention(self):
        test_model_dim = 512
        ground_truth = (1, test_model_dim, 17)
        test_num_heads = 8
        test_tensor = torch.zeros(ground_truth)
        test_attention = SelfAttention(num_heads=test_num_heads, model_dim=test_model_dim)
        result = test_attention.forward(test_tensor)
        assert result.shape == ground_truth

        test_attention = SelfAttention(num_heads=test_num_heads, model_dim=test_model_dim, masked=True)
        result = test_attention.forward(test_tensor)
        assert result.shape == ground_truth

    def test_cross_attention(self):
        test_model_dim = 512
        ground_truth = (1, test_model_dim, 17)
        test_num_heads = 8
        test_tensor = torch.zeros(ground_truth)
        test_attention = CrossAttention(num_heads=test_num_heads, model_dim=test_model_dim)
        test_attention.encoder_tokens = torch.zeros(ground_truth)
        result = test_attention.forward(test_tensor)
        assert result.shape == ground_truth

    def test_positional_encoding(self):
        model_dim = 512
        ground_truth = (1, model_dim, 42)

        data = torch.zeros(ground_truth)
        encoding = PositionalEncoding(model_dim=model_dim)
        result = encoding(data)

        assert result.shape == ground_truth

        model_dim = 4
        token_count = 4

        ground_truth = torch.tensor([[0.0000, 0.8410, 0.9090, 0.1410],
                                     [1.0000, 0.5400, -0.4160, -0.9900],
                                     [0.0000, 0.1000, 0.1990, 0.2960],
                                     [1.0000, 0.9950, 0.9800, 0.9550]])

        data = torch.zeros((1, model_dim, token_count))
        encoding = PositionalEncoding(model_dim=model_dim)
        result = torch.round(encoding(data), decimals=3)

        assert torch.equal(result[0], ground_truth)

    def test_feed_forward(self):
        model_dim = 512
        ground_truth = (1, model_dim, 17)

        data = torch.zeros(ground_truth)
        ff = FeedForward(model_dim=model_dim)
        result = ff(data)
        assert result.shape == ground_truth

    def test_transpose(self):
        dim = 8
        tokens = 4
        heads = 2
        data = torch.zeros((1, heads * dim, tokens))
        # keys = torch.zeros((tokens, heads * dim))

        att = SelfAttention(heads, dim)
        data = att.reshape_for_multi_head(data)
        assert data.shape == (1, heads, dim, tokens)

        # keys = att.transpose_for_multi_head(keys)

        # torch.matmul(data, keys.transpose(-1, -2))
