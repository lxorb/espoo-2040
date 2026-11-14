import json
import os

import numpy as np
import torch
from torch import nn

from .policy import AF, GF, HERE, PF, joints, jmask, obs


class Net(nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.heads = heads
        L = nn.Linear
        for i, f in enumerate((GF, AF, PF)):
            setattr(self, f"e{i}", L(f, d))
            setattr(self, f"f{i}", L(d, d))
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(n1=nn.LayerNorm(d), qkv=L(d, 3 * d), o=L(d, d), n2=nn.LayerNorm(d),
                                                       m1=L(d, 2 * d), m2=L(2 * d, d))) for _ in range(layers))
        self.hg, self.hp = L(d, 7), L(d, 1)
        for h in (self.hg, self.hp):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, g, a, p):
        x = torch.cat([getattr(self, f"f{i}")(torch.relu(getattr(self, f"e{i}")(v))) for i, v in enumerate((g, a, p))], 1)
        B, T, d = x.shape
        for b in self.blocks:
            h = b["qkv"](b["n1"](x))
            q, k, v = (h[..., i::3].reshape(B, T, self.heads, -1).transpose(1, 2) for i in range(3))
            o = torch.softmax(q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5, -1) @ v
            x = x + b["o"](o.transpose(1, 2).reshape(B, T, d))
            x = x + b["m2"](torch.relu(b["m1"](b["n2"](x))))
        hg = self.hg(x[:, 0])
        return hg[:, :3], hg[:, 3:6], hg[:, 6], self.hp(x[:, 1 + a.shape[1]:])[..., 0]

    @staticmethod
    def npz(path):
        W = np.load(path)
        return {("blocks." if k[0].isdigit() else "") + k[:-2] + (".weight" if k.endswith(".w") else ".bias"): W[k] for k in W.files if k != "layers"}

    def export(self, path):
        W = {k.replace("blocks.", "").replace(".weight", ".w").replace(".bias", ".b"): v.detach().numpy() for k, v in self.state_dict().items()}
        np.savez(path, layers=len(self.blocks), **W)


def jlogits(J, A, C, B):
    B = torch.cat([B, torch.zeros_like(B[:, :1])], 1)
    J = [torch.from_numpy(j) for j in J]
    return A[:, J[0]] + C[:, J[3]] + B[:, J[1]] + B[:, J[2]]


def places(c):
    s, zs = c.s, json.loads((c.data / "zones.json").read_text())
    ix = {z["id"]: i for i, z in enumerate(zs["zones"])}
    mid = (c.nodes[s["a"].astype(int)] + c.nodes[s["b"].astype(int)]) / 2

    def at(q):
        pts = [g for e in q.get("new", []) for g in e["geom"]] + [g for e in q.get("pt", []) for g in e["stops"]]
        pts += q.get("park", {}).get("area", []) + ([q["pnr"]["at"]] if "pnr" in q else [])
        osm = [o for k in ("mod", "works") for e in q.get(k, []) for o in e["osm"]] + q.get("toll", {}).get("osm", [])
        pts += mid[np.isin(s["osm"], osm)].tolist()
        return np.mean(pts, 0).round(1).tolist() if pts else [0.0, 0.0]

    area = {a["name"]: np.average(c.zxy[[ix[z] for z in a["zones"]]], 0, c.pop0[[ix[z] for z in a["zones"]]] + 1).round(1).tolist()
            for a in zs["growth"]["areas"]}
    return dict(projects={k: at(c.cat[k]) for k in c.menu}, areas=area)


class Agent:
    def __init__(self, c, net):
        self.menu, self.net, self.J = list(c.menu), net, joints(len(c.menu))
        self.proj = [dict(id=k, **{f: c.cat[k].get(f, 0) for f in ("kind", "cost", "years", "opex")}) for k in self.menu]

    def look(self, c):
        m, jam = np.array(c.mask(), bool), {q["id"]: q["jam"] for q in c.catalogue() if not os.environ.get("RL_NOJAM")}
        return obs(c.state(), [dict(q, jam=jam.get(q["id"], 0.0)) for q in self.proj], c.years), jmask(self.J, m)

    def eval(self, ob):
        with torch.no_grad():
            A, C, v, B = self.net(*(torch.from_numpy(x)[None] for x in ob))
            return jlogits(self.J, A, C, B)[0].numpy().astype(float), float(v[0])


def batch(xs):
    return [torch.from_numpy(np.stack([x[0][i] for x in xs])) for i in range(3)]


if __name__ == "__main__":
    from sim import City
    (HERE / "places.json").write_text(json.dumps(places(City(0))))
