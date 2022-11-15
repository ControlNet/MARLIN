import torch
from torch import Tensor, nn

from .util import Shape


class PositionalEmbedding(nn.Module):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = nn.Parameter(torch.zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value


class SinCosPositionalEmbedding(PositionalEmbedding):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode='trunc') / d_hid)

        sinusoid_table = torch.stack([get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()
