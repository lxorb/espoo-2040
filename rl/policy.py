import json
from pathlib import Path

import numpy as np

KINDS = ["road", "lanes", "junction", "buslane", "rail", "bus", "toll", "parking", "pnr"]
GF, AF, PF = 11, 8, len(KINDS) + 15
WEIGHTS = Path(__file__).with_name("policy.npz")
PLACES = json.loads(Path(__file__).with_name("places.json").read_text()) if Path(__file__).with_name("places.json").exists() else {"projects": {}, "areas": {}}


def joints(n):
    J = [(m, n, n) for m in range(3)] + [(m, i, n) for m in range(3) for i in range(n)]
    J += [(m, i, j) for m in range(3) for i in range(n) for j in range(i + 1, n)]
    J = np.array(J)
    return J[:, 0], J[:, 1], J[:, 2], (J[:, 1] < n).astype(int) + (J[:, 2] < n)


def jmask(J, mask):
    ok = np.r_[np.asarray(mask, bool), True]
    return ok[J[1]] & ok[J[2]]


def decode(J, menu, a):
    return [{"maintain": int(J[0][a])}] + [{"build": menu[i]} for i in (J[1][a], J[2][a]) if i < len(menu)]


def encode(J, menu, acts):
    m = next((a["maintain"] for a in acts if "maintain" in a), 1)
    b = sorted(menu.index(a["build"]) for a in acts if "build" in a) + [len(menu)] * 2
    return int(np.flatnonzero((J[0] == m) & (J[1] == b[0]) & (J[2] == b[1]))[0])


def obs(st, projects, mask, years=15, places=None, jam=None):
    places = places or PLACES
    t, inc, k = st["step"], max(st["income"], 1e-6), max(st["step"], 1)
    cond = np.r_[st["cond"], 1.0, 1.0][:2]
    g = np.array([[t / years, max(years - t, 0) / years, st["budget"] / inc / 5, st["committed"] / inc / 10, st["opex"] / inc,
                   st["car"] or 0.5, *cond, np.log1p(st["jam"] or 0) / 20, np.log1p(st["last"] or 0) / 20, t >= years]], np.float32)
    ar = st["areas"]
    rp = np.array([r["pop"] / max(r["pop0"], 1) for r in ar])
    rj = np.array([r["jobs"] / max(r["jobs0"], 1) for r in ar])
    rate = ((rp * rj) ** (0.5 / k) - 1) * 20
    axy = np.array([places["areas"].get(r["name"], (0.0, 0.0)) for r in ar], float).reshape(-1, 2)
    a = np.stack([np.log1p([r["pop0"] for r in ar]) / 12, np.log1p([r["jobs0"] for r in ar]) / 12, rp - 1, rj - 1,
                  (rp ** (1 / k) - 1) * 20, (rj ** (1 / k) - 1) * 20, axy[:, 0] / 2e4, axy[:, 1] / 2e4], 1).astype(np.float32).reshape(-1, AF)
    pxy = np.array([places["projects"].get(q["id"], (0.0, 0.0)) for q in projects], float)
    w = np.exp(-np.linalg.norm(pxy[:, None] - axy[None], axis=2) / 3000) if len(ar) else np.zeros((len(projects), 1))
    p = np.zeros((len(projects), PF), np.float32)
    jam = np.zeros(len(projects)) if jam is None else np.asarray(jam, float)
    for i, q in enumerate(projects):
        p[i, KINDS.index(q["kind"]) if q["kind"] in KINDS else 0] = 1
        p[i, len(KINDS):] = [np.log1p(q["cost"]) / 7, q["cost"] / inc, q["years"] / 8, q.get("opex", 0) / inc * 10,
                             q["id"] in st["works"], st["works"].get(q["id"], 0) / 8, q["id"] in st["built"],
                             pxy[i, 0] / 2e4, pxy[i, 1] / 2e4, (w[i] * rate).sum(), (w[i] * rate).max(initial=0),
                             (w[i] * (np.sqrt(rp * rj) - 1)).sum(), w[i].sum(), jam[i] * 10, jam[i] / max(jam.max(), 1e-9)]
    return tuple(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (g, a, p))


def ln(x, w, b):
    m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def forward(W, g, a, p, heads=4):
    x = np.concatenate([np.maximum(v @ W[f"e{i}.w"].T + W[f"e{i}.b"], 0) @ W[f"f{i}.w"].T + W[f"f{i}.b"] for i, v in enumerate((g, a, p))])
    for l in range(int(W["layers"])):
        h = ln(x, W[f"{l}.n1.w"], W[f"{l}.n1.b"]) @ W[f"{l}.qkv.w"].T + W[f"{l}.qkv.b"]
        q, k, v = (h[:, i::3].reshape(len(x), heads, -1).transpose(1, 0, 2) for i in range(3))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
        s = np.exp(s - s.max(-1, keepdims=True))
        o = (s / s.sum(-1, keepdims=True) @ v).transpose(1, 0, 2).reshape(len(x), -1)
        x = x + o @ W[f"{l}.o.w"].T + W[f"{l}.o.b"]
        h = np.maximum(ln(x, W[f"{l}.n2.w"], W[f"{l}.n2.b"]) @ W[f"{l}.m1.w"].T + W[f"{l}.m1.b"], 0)
        x = x + h @ W[f"{l}.m2.w"].T + W[f"{l}.m2.b"]
    hg = x[0] @ W["hg.w"].T + W["hg.b"]
    hp = x[1 + len(a):] @ W["hp.w"].T + W["hp.b"]
    return hg[:3], hg[3:6], hg[6], hp[:, 0]


def joint(J, A, C, B):
    B = np.r_[B, 0]
    return A[J[0]] + C[J[3]] + B[J[1]] + B[J[2]]


class Policy:
    def __init__(self, path=WEIGHTS, check=False, maintain=2):
        self.W, self.check, self.maintain = dict(np.load(path)), check, maintain

    def __call__(self, st, projects, mask, forecast=None, k=4, years=15):
        menu = [q["id"] for q in projects]
        J = joints(len(menu))
        A, C, _, B = forward(self.W, *obs(st, projects, mask, years, None, [q.get("jam", 0.0) for q in projects]))
        lg = np.where(jmask(J, mask), joint(J, A, C, B), -np.inf)
        top = np.argsort(-lg)[: k if self.check and forecast else 1]
        acts = decode(J, menu, min(top, key=lambda i: forecast(decode(J, menu, i))) if len(top) > 1 else top[0])
        return [{"maintain": self.maintain} if "maintain" in a and self.maintain is not None else a for a in acts]
