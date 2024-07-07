"""Module to implement the Vaswani et al. 2017 Transformer with slight adjustments."""
import numpy as np
import torch


class Transformer(torch.nn.Module):
    """Implements transformer architecture as described in Vaswani et al. 2017.
    Contrary to the original paper, this implementation uses batch norm instead of LayerNorm.
    LayerNorm has the drawback, that it forces a fixed number of input tokens, because of the learned bias
    weights and running means."""

    def __init__(self, word_token_dim, num_layers: int = 6, model_dim: int = 512,
                 num_heads: int = 8, linear_hidden_dim=2048):
        super().__init__()
        self.encoder = Encoder(num_layers, model_dim, word_token_dim, num_heads, linear_hidden_dim)
        self.decoder = Decoder(num_layers, model_dim, word_token_dim, num_heads, linear_hidden_dim)
        self.model_dim = model_dim
        self.num_heads: int = num_heads

    def forward(self, encoder_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        """Executes encoder and decoder."""
        encoder_result = self.encoder(encoder_tokens)
        decoder_result = self.decoder(encoder_result, decoder_tokens)
        return decoder_result  # type:ignore


class Encoder(torch.nn.Module):
    """Implements encoder with multiple layers consisting of attention and feed forward modules, as well as embedding
    and positional encoding."""

    def __init__(self, num_layers: int, model_dim: int, word_token_dim: int, num_heads: int, linear_hidden_dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Embedding(model_dim, word_token_dim))
        self.layers.append(PositionalEncoding(model_dim))
        for _ in range(num_layers):
            self.layers.append(SelfAttention(num_heads, model_dim))
            self.layers.append(FeedForward(model_dim, linear_hidden_dim))

    def forward(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes encoder layers
        """
        for layer in self.layers:
            encoder_tokens = layer(encoder_tokens)
        return encoder_tokens


class Decoder(torch.nn.Module):
    """Implements decoder with masked self attention, as well as multiple layers consisting of cross attention and
    feed forward modules, as well as embedding and positional encoding for the output sequence."""

    def __init__(self, num_layers: int, model_dim: int, word_token_dim: int, num_heads: int, linear_hidden_dim: int):
        super().__init__()

        self.cross_attentions = []

        self.layers = torch.nn.ModuleList()
        self.layers.append(Embedding(model_dim, word_token_dim))
        self.layers.append(PositionalEncoding(model_dim))
        for _ in range(num_layers):
            self.layers.append(SelfAttention(num_heads, model_dim, True))
            cross_attention = CrossAttention(num_heads, model_dim)
            self.cross_attentions.append(cross_attention)
            self.layers.append(cross_attention)
            self.layers.append(FeedForward(model_dim, linear_hidden_dim))
        self.layers.append(torch.nn.Conv1d(model_dim, word_token_dim, 1))
        self.layers.append(torch.nn.Softmax(dim=1))

    def set_encoder_tokens(self, encoder_tokens: torch.Tensor):
        """Sets encoder token attribute of all CrossAttention modules, as all CrossAttentions use the same encoder
        output as key and value."""
        for cross_attention in self.cross_attentions:
            cross_attention.encoder_tokens = encoder_tokens  # type:ignore

    def forward(self, encoder_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        """
        Executes decoder layers
        """
        self.set_encoder_tokens(encoder_tokens)
        for layer in self.layers:
            decoder_tokens = layer(decoder_tokens)
        return decoder_tokens


class Embedding(torch.nn.Module):
    """Implements fully connected layer for converting word tokens into vectors of model dimension.
    """

    def __init__(self, model_dim, token_dim):
        super().__init__()
        self.linear = torch.nn.Conv1d(token_dim, model_dim, 1)

    def forward(self, word_tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts word tokens into vectors of model dimension.
        """
        return self.linear(word_tokens)  # type:ignore


class PositionalEncoding(torch.nn.Module):
    """Implements cosine function based positional encoding like in Vasvani et al. 2017"""

    def __init__(self, model_dim: int, wavelength_factor=10000):
        super().__init__()
        self.model_dim = model_dim
        self.wavelength_factor = wavelength_factor

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Computes positional encodings depenging on the number of input tokens, as well as the model dimension.
        The encoding are then added to the input tokens. Implementation was inspired by
        https://pub.aimind.so/creating-sinusoidal-positional-embedding-from-scratch-in-pytorch-98c49e153d6"""
        token_count = tokens.shape[-1]
        assert token_count % 2 == 0, "positional embeddings cannot be applied to an odd number of input tokens."
        positions = torch.arange(token_count).unsqueeze_(1)
        denominators = torch.pow(self.wavelength_factor, torch.arange(self.model_dim // 2) / self.model_dim)
        tokens[:, 0::2, :] += torch.sin(positions / denominators).T
        tokens[:, 1::2, :] += torch.cos(positions / denominators).T
        return tokens


class SelfAttention(torch.nn.Module):
    """Implements multi head scaled dot product attention submodule."""

    def __init__(self, num_heads: int, model_dim: int, masked=False):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.masked = masked

        # As stated in Vaswani et al. 2017 1x1 convolutions are equivalent to computing the output of a linear layer
        # with a series of input tokens.
        self.linear_query = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.linear_key = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.linear_value = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.linear_output = torch.nn.Conv1d(model_dim * num_heads, model_dim, 1)

        self.bn = torch.nn.BatchNorm1d(model_dim)

    def reshape_for_multi_head(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transpose vectors of size dim*num_heads to matrices of shape [dim, num_heads].
        Implementation from https://github.com/Beckschen/TransUNet.
        """
        new_x_shape = (tokens.shape[0], tokens.shape[-1], self.num_heads, self.model_dim)
        tokens = tokens.view(*new_x_shape)
        return tokens.permute(0, 2, 3, 1)

    def compute_attention(self, key, query, value):
        """Computes attention mechanism by multiplying key and query matrices and apply the resulting probabilities
        to the value matrix."""

        if self.masked:
            mask = torch.ones((key.shape[-1], key.shape[-1]), dtype=bool)
            mask = ~torch.tril(mask).T.repeat(key.shape[0], self.num_heads, 1, 1)
        else:
            mask = ~torch.ones((key.shape[-1], key.shape[-1]), dtype=bool).repeat(key.shape[0], self.num_heads, 1, 1)

        key = self.reshape_for_multi_head(self.linear_key(key))
        query = self.reshape_for_multi_head(self.linear_query(query))
        value = self.reshape_for_multi_head(self.linear_value(value))
        result = torch.matmul(
            torch.nn.functional.softmax(torch.masked_fill(torch.matmul(query.permute(0, 1, 3, 2), key) /
                                                          np.sqrt(self.model_dim), mask, float('-inf')), dim=1),
            value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        result = result.reshape(result.shape[0], result.shape[1] * result.shape[2], result.shape[3])
        result = self.linear_output(result)
        return result

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes attention, batch norm and implements skip attention.
        """
        identity = tokens.clone()
        result = self.compute_attention(tokens, tokens, tokens)
        result = self.bn(result)
        return result + identity  # type:ignore


class CrossAttention(SelfAttention):
    """Implements Cross Attention between encoder key and query and decoder value"""

    def __init__(self, num_heads: int, model_dim: int):
        super().__init__(num_heads, model_dim)
        self.encoder_tokens = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes cross attention, batch norm and implements skip attention, which copies the decoder token.

        Args:
            tokens: decoder tokens, that are fed to the attention mechanism as value tokens.
        """
        identity = tokens.clone()
        result = self.compute_attention(self.encoder_tokens, self.encoder_tokens, tokens)
        result = self.bn(result)
        return result + identity  # type:ignore


class FeedForward(torch.nn.Module):
    """Implements feed-forward layer as described in Vaswani et al. 2017"""

    def __init__(self, model_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.linear_in = torch.nn.Conv1d(model_dim, hidden_dim, 1)
        self.linear_out = torch.nn.Conv1d(hidden_dim, model_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes cross attention, batch norm and implements skip attention, which copies the decoder token..
        """
        identity = tokens.clone()
        result = self.linear_in(tokens)
        result = torch.nn.functional.relu(result)
        result = self.linear_out(result)
        return result + identity  # type:ignore
