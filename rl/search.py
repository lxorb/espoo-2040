import numpy as np

from .policy import decode


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def sigma(q, n=1):
    return (50 + n) * 0.1 * (q - q.min()) / max(q.max() - q.min(), 1e-8)


class Search:
    def __init__(self, ag, base, scale, n=16, depth=4):
        self.ag, self.base, self.scale, self.n, self.depth = ag, base, scale, n, depth

    def leaf(self, c, acts):
        f, q = c.fork(), 0.0
        for _ in range(self.depth):
            if f.done:
                return q
            y = f.state()["step"]
            q += np.nan_to_num((self.base[y] - f.step(acts)["delay"]) / self.scale)
            acts = []
        return q + (0.0 if f.done else self.ag.eval(self.ag.look(f)[0])[1])

    def __call__(self, c, gscale, rng):
        ob, jm = self.ag.look(c)
        lg, v = self.ag.eval(ob)
        lg = np.where(jm, lg, -np.inf)
        g = gscale * rng.gumbel(size=lg.size)
        top = np.argsort(-(g + lg))[: min(self.n, int(jm.sum()))]
        q = np.full(lg.size, np.nan)
        for a in top:
            q[a] = self.leaf(c, decode(self.ag.J, self.ag.menu, a))
        pi = softmax(lg)
        vm = (v + top.size * (pi[top] * q[top]).sum() / max(pi[top].sum(), 1e-12)) / (1 + top.size)
        s = sigma(np.where(jm, np.where(np.isnan(q), vm, q), vm))
        a = int(top[np.argmax((g + lg + s)[top])])
        return a, softmax(lg + s), ob, jm
