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


class Encoder(torch.nn.Module):
    """Implements encoder with multiple layers consisting of attention and feed forward modules, as well as embedding
    and positional encoding."""

    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Embedding())
        self.layers.append(PositionalEncoding())
        for i in range(num_layers):
            self.layers.append(Attention())
            self.layers.append(FeedForward())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class Decoder(torch.nn.Module):
    """Implements decoder with mask self attention, as well as multiple layers consisting of cross attention and
    feed forward modules, as well as embedding and positional encoding for the output sequence."""

    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Embedding())
        self.layers.append(PositionalEncoding())
        for i in range(num_layers):
            self.layers.append(Attention())
            self.layers.append(FeedForward())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class Embedding(torch.nn.Module):
    """Implements fully connected layer for converting word tokens into vectors of model dimension.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Conv1d(input_size, output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class PositionalEncoding(torch.nn.Module):
    """Implements cosine function based positional encoding like in Vasvani et al. 2017"""

    def __init__(self):
        super().__init__()
        # sin
        # cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass


class Attention(torch.nn.Module):
    """Implements multi head scaled dot product attention."""

    def __init__(self, num_heads: int, model_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.conv_query = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.conv_key = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.conv_value = torch.nn.Conv1d(model_dim, num_heads * model_dim, 1)
        self.conv_output = torch.nn.Conv1d(model_dim, model_dim, 1)

    def transpose_for_multi_head(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transpose vectors of size dim*num_heads to matrices of shape [dim, num_heads].
        Implementation from https://github.com/Beckschen/TransUNet.
        """
        new_x_shape = inputs.size()[:-1] + (self.num_heads, self.model_dim)
        inputs = inputs.view(*new_x_shape)
        return inputs.permute(1, 0, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear
        # transpose_for_multi_head()
        # scaled dot product: softmax(Q*K/sqrt(input_dim) * V)
        # concat (multi head)
        # linear
        # TODO: implement
        pass


class FeedForward(torch.nn.Module):
    """Implements feed-forward layer as described in Vaswani et al. 2017"""

    def __init__(self, model_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        # linear input to hidden + ReLu; hidden to output do all with 1x1 conv over whole matrix
