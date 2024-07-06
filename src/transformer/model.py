import numpy as np
import torch


class Transformer(torch.nn.Module):
    """Implements transformer architecture as described in Vaswani et al. 2017"""

    def __init__(self, num_layers: int = 6, dim: int = 512, ff_dim: int = 2048, num_heads: int = 8):
        super().__init__()
        # TODO: yml config?
        self.encoder = Encoder(num_layers)
        self.decoder = Decoder(num_layers)
        self.dim = dim
        self.ff_dim = ff_dim
        self.num_heads: int = num_heads

    def forward(self, encoder_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class Encoder(torch.nn.Module):
    """Implements encoder with multiple layers consisting of attention and feed forward modules, as well as embedding
    and positional encoding."""

    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Embedding())
        self.layers.append(PositionalEncoding())
        for i in range(num_layers):
            self.layers.append(SelfAttention())
            self.layers.append(FeedForward())

    def forward(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        """this is a forward pass"""
        for layer in self.layers:
            encoder_tokens = layer(encoder_tokens)
        encoder_output = encoder_tokens
        return encoder_output


class Decoder(torch.nn.Module):
    """Implements decoder with mask self attention, as well as multiple layers consisting of cross attention and
    feed forward modules, as well as embedding and positional encoding for the output sequence."""

    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Embedding())
        self.layers.append(PositionalEncoding())
        # TODO: implement Masked Multi-Head Attention
        for i in range(num_layers):
            self.layers.append(SelfAttention())  # TODO add cross-Attention
            self.layers.append(FeedForward())

    def forward(self, decoder_tokens: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # TODO: implement

        CrossAttention.encoder_output = encoder_output
        for layer in self.layers:
            # implement me
            pass
        CrossAttention.encoder_output = None
        pass


class Embedding(torch.nn.Module):
    """Implements fully connected layer for converting word tokens into vectors of model dimension.
    """

    def __init__(self, token_dim, model_dim):
        super().__init__()
        self.linear = torch.nn.Conv1d(token_dim, model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class PositionalEncoding(torch.nn.Module):
    """Implements cosine function based positional encoding like in Vasvani et al. 2017"""

    def __init__(self):
        super().__init__()
        # TODO: implement me
        # sin
        # cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class SelfAttention(torch.nn.Module):
    """Implements multi head scaled dot product attention."""

    def __init__(self, num_heads: int, model_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.conv_query = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.conv_key = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.conv_value = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.conv_output = torch.nn.Conv1d(model_dim * num_heads, model_dim, 1)

    def transpose_for_multi_head(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transpose vectors of size dim*num_heads to matrices of shape [dim, num_heads].
        Implementation from https://github.com/Beckschen/TransUNet.
        """
        new_x_shape = tokens.shape[-1:] + (self.num_heads, self.model_dim)
        tokens = tokens.view(*new_x_shape)
        return tokens.permute(1, 2, 0)

    def compute_attention(self, key, query, value):
        assert len(key.shape) == 2
        assert len(query.shape) == 2
        assert len(value.shape) == 2
        key = self.transpose_for_multi_head(self.conv_key(key))
        query = self.transpose_for_multi_head(self.conv_query(query))
        value = self.transpose_for_multi_head(self.conv_value(value))
        out = torch.matmul(
            torch.nn.functional.softmax(torch.matmul(query.permute(0, 2, 1), key) / np.sqrt(self.model_dim)),
            value.permute(0, 2, 1)).permute(0, 2, 1)
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])
        out = self.conv_output(out)
        return out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.compute_attention(tokens, tokens, tokens)
        return out


class CrossAttention(SelfAttention):
    """Implements Cross Attention between encoder key and query and decoder value"""

    def __init__(self, num_heads: int, model_dim: int):
        super().__init__(num_heads, model_dim)
        self.encoder_output = None

    def forward(self, decoder_tokens: torch.Tensor) -> torch.Tensor:
        out = self.compute_attention(self.encoder_output, self.encoder_output, decoder_tokens)
        return out


class FeedForward(torch.nn.Module):
    """Implements feed-forward layer as described in Vaswani et al. 2017"""

    # TODO: implement me

    def __init__(self, model_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        # linear input to hidden + ReLu; hidden to output do all with 1x1 conv1D over whole matrix
