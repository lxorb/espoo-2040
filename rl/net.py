import numpy as np
import torch
from torch import nn

from .env import GF, PF, SF


class Net(nn.Module):
    def __init__(self, d=96, layers=3):
        super().__init__()
        self.inp = nn.ModuleList(nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d)) for f in (GF, PF, SF))
        self.typ = nn.Parameter(torch.randn(3, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.out = nn.ModuleList(nn.Linear(d, k) for k in (2, 1, 2))
        for o in self.out:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

    def forward(self, g, p, s):
        x = torch.cat([f(v) + t for f, v, t in zip(self.inp, (g, p, s), self.typ)], 1)
        x = self.enc(x)
        n = p.shape[1]
        h = [o(v) for o, v in zip(self.out, (x[:, 0], x[:, 1:1 + n], x[:, 1 + n:]))]
        return torch.cat([h[0][:, :1], h[1][..., 0], h[2][..., 0], h[2][..., 1]], 1), h[0][:, 1]


def infer(net, obs):
    with torch.no_grad():
        lg, v = net(*(torch.from_numpy(x)[None] for x in obs))
    return lg[0].numpy().astype(np.float64), float(v[0])
