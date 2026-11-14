import copy
import hashlib
from pathlib import Path

import numpy as np

from sim import City
from sim import params as P

K, M = 16, 4
KINDS = ["road", "lanes", "junction", "buslane", "rail", "bus", "toll", "parking", "pnr"]
CLS = list(P.CAP_LANE)
GF, PF, SF = 12, len(KINDS) + 9, len(CLS) + 10


def version():
    root = Path(__file__).resolve().parent.parent
    files = sorted(root.glob("sim/*.py")) + sorted((root / "data" / "out").glob("*")) + [root / "data" / "catalogue.json"]
    return hashlib.sha1(b"".join(f.read_bytes() for f in files if f.is_file())).hexdigest()[:8]


def fork(c):
    f = copy.copy(c)
    f.s, f.recs, f.rp, f.rj, f.who, f.gap = copy.deepcopy(c.s), [], c.mp, c.mj, "forecast", c.fast
    return f


def levels(q):
    return len((q.get("toll") or q.get("park") or {"eur": [0]})["eur"])


class Game:
    def __init__(self, seed, full=True, years=15, runout=5):
        self.c = City(seed, years, runout)
        recs = self.c.forecast({}, years + runout if full else 1)
        self.base, self.rec = [r["delay"] for r in recs], recs[0]
        self.scale = self.base[0] / 10
        self.pj = [(p, l) for p, q in self.c.cat.items() for l in range(levels(q))]

    def reward(self, c, rec):
        return (self.base[c.s["year"] - 1] - rec["delay"]) / self.scale

    def view(self, c, rec, acts):
        s, f = c.s, fork(c)
        for a in acts:
            f._act(a, [])
        y, cash = s["year"], f.s["cash"]
        ids, inv = np.unique(s["seg"], return_inverse=True)
        n = ids.size

        def agg(fn, v, init):
            o = np.full(n, init, float)
            fn.at(o, inv, v)
            return o

        dl = np.bincount(inv, rec["flow"] * np.maximum(rec["time"] / 60 - c._t0(), 0), n)
        cond, vc, rt = agg(np.minimum, s["cond"], 1), agg(np.maximum, rec["flow"] / s["cap"], 0), agg(np.maximum, s["retime"], 0)
        lk = np.bincount(inv, s["len"] * s["lanes"], n) / 1000
        xy = np.stack([agg(np.add, c.nodes[s["a"].astype(int), k], 0) for k in (0, 1)], 1) / np.bincount(inv, minlength=n)[:, None]
        cl = np.zeros(n, int)
        cl[inv] = [CLS.index(k) for k in s["cls"]]
        hot = np.argsort(-dl)
        pick = np.array(list(dict.fromkeys([*np.argsort(cond)[:K], *hot[:4 * K]]))[: 2 * K])
        sel = {(k, v) for a in acts for k, v in a.items() if k in ("resurface", "retime")}
        rs = np.array([("resurface", ids[i]) in sel for i in pick])
        ts = np.array([("retime", ids[i]) in sel for i in pick])
        S = np.zeros((2 * K, SF), np.float32)
        S[np.arange(2 * K), cl[pick]] = 1
        S[:, len(CLS):] = np.stack([cond[pick], np.log1p(lk[pick]), dl[pick] / dl.sum() * 20, vc[pick], rt[pick] * 10,
                                    xy[pick, 0] / 2e4, xy[pick, 1] / 2e4, rs, ts, (cond[pick] - P.COND_MIN) * 5], 1)
        st, cat = c.state(), {p["id"]: p for p in f.catalogue()}
        chosen = {a.get("build") for a in acts}
        open_ = y < c.years and len(acts) < M
        Pm = np.zeros((len(self.pj), PF), np.float32)
        pm = np.zeros(len(self.pj), bool)
        for i, (p, l) in enumerate(self.pj):
            q = c.cat[p]
            Pm[i, KINDS.index(q["kind"])] = 1
            Pm[i, len(KINDS):] = [np.log1p(q["cost"]) / 7, q["cost"] / max(cash, 1) / 10, q["years"] / 8, q.get("opex", 0) / 10,
                                  l / levels(q), p in st["works"], st["works"].get(p, 0) / 8, p in st["built"], p in chosen]
            pm[i] = open_ and p in cat and cat[p]["afford"] and p not in chosen
        e = max(y - 1, 0)
        g = np.array([y / c.years, max(c.years - y, 0) / c.years, cash / 200, st["committed"] / 200, st["opex"] / 20,
                      s["car"] or 0.5, (s["fp"] * c.pop0).sum() / c.pop0.sum() - 1, ((1 + c.mp) ** e * c.pop0).sum() / c.pop0.sum() - 1,
                      (s["fj"] * c.jobs0).sum() / c.jobs0.sum() - 1, rec["delay"] / self.base[0], len(acts) / M, y >= c.years], np.float32)
        mask = np.r_[True, pm, open_ & ~rs & (P.RESURF * lk[pick] <= cash), open_ & ~ts & (P.RETIME_COST <= cash)]
        acts_ = [None] + [{"build": p, "level": l} for p, l in self.pj]
        acts_ += [{"resurface": int(ids[i])} for i in pick] + [{"retime": int(ids[i])} for i in pick]
        return (g[None], Pm, S), mask, acts_
