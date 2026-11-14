import math

import numpy as np

from .env import fork
from .net import infer


def halving(k, n):
    if k <= 1:
        return list(range(n))
    seq, vis, m, lg = [], [0] * k, k, math.ceil(math.log2(k))
    while len(seq) < n:
        for _ in range(max(1, int(n / (lg * m)))):
            seq += vis[:m]
            vis[:m] = [v + 1 for v in vis[:m]]
        m = max(2, m // 2)
    return seq[:n]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Node:
    def __init__(self, lg, mask, acts, v):
        self.lg, self.mask, self.acts, self.v = np.where(mask, lg, -np.inf), mask, acts, v
        self.N, self.W, self.kids = np.zeros(lg.size), np.zeros(lg.size), {}

    def sigma(self):
        pi, vis = softmax(self.lg), self.N > 0
        n = self.N.sum()
        vm = (self.v + n * (pi[vis] * self.W[vis] / self.N[vis]).sum() / max(pi[vis].sum(), 1e-12)) / (1 + n) if n else self.v
        q = np.where(vis, self.W / np.maximum(self.N, 1), vm)
        lo, hi = q[self.mask].min(), q[self.mask].max()
        return (50 + self.N.max()) * 0.1 * (q - lo) / max(hi - lo, 1e-8)


class Search:
    def __init__(self, game, net, c, rec, depth=1):
        self.game, self.net, self.c, self.rec, self.depth, self.cache = game, net, c, rec, depth, {}

    def leaf(self, acts):
        k = frozenset(tuple(sorted(a.items())) for a in acts)
        if k not in self.cache:
            f = fork(self.c)
            r = f.step(acts)
            q = self.game.reward(f, r)
            for _ in range(self.depth - 1):
                if not f.done:
                    r = f.step()
                    q += self.game.reward(f, r)
            self.cache[k] = q + (0.0 if f.done else infer(self.net, self.game.view(f, r, [])[0])[1])
        return self.cache[k]

    def expand(self, acts, value=True):
        obs, mask, cand = self.game.view(self.c, self.rec, acts)
        lg, _ = infer(self.net, obs)
        nd = Node(lg, mask, acts, self.leaf(acts) if value else 0.0)
        nd.obs, nd.cand = obs, cand
        return nd

    def sim(self, nd, a):
        path = []
        while True:
            path.append((nd, a))
            if a == 0:
                val = nd.v
                break
            if a not in nd.kids:
                nd.kids[a] = self.expand(nd.acts + [nd.cand[a]])
                val = nd.kids[a].v
                break
            nd = nd.kids[a]
            sc = softmax(nd.lg + nd.sigma()) - nd.N / (1 + nd.N.sum())
            a = int(np.argmax(np.where(nd.mask, sc, -np.inf)))
        for p, b in path:
            p.N[b] += 1
            p.W[b] += val

    def run(self, acts, n, k, gscale, rng):
        if n == 0:
            nd = self.expand(acts, False)
            return int(np.argmax(nd.lg)), None, nd
        nd = self.expand(acts)
        if nd.mask.sum() == 1:
            return 0, nd.mask.astype(float), nd
        g = np.where(nd.mask, gscale * rng.gumbel(size=nd.lg.size), 0)
        top = np.r_[0, [i for i in np.argsort(-(g + nd.lg)) if i][: min(k, int(nd.mask.sum())) - 1]].astype(int)
        for v in halving(top.size, n):
            c = top[nd.N[top] == v]
            self.sim(nd, int(c[np.argmax((g + nd.lg + nd.sigma())[c])]))
        s = nd.sigma()
        sc = np.where(nd.N == nd.N.max(), g + nd.lg + s, -np.inf)
        return int(np.argmax(np.where(nd.mask, sc, -np.inf))), softmax(nd.lg + s), nd


def year(game, net, c, rec, n=8, k=4, gscale=1.0, rng=None, depth=1):
    s, acts, out = Search(game, net, c, rec, depth), [], []
    while True:
        a, pi, nd = s.run(acts, n, k, gscale, rng or np.random.default_rng())
        if pi is not None:
            out.append((nd.obs, nd.mask, pi))
        if a == 0:
            return acts, out
        acts = acts + [nd.cand[a]]
