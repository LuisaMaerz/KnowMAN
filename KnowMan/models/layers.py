import torch
from torch import nn


class TransformerHeadFeature(nn.Module):
    def __init__(self, config, drop_out, out_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(drop_out)
        self.out_proj = nn.Linear(config.hidden_size, out_size)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
