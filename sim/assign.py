import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from . import params as P


@njit(cache=True)
def _tree(src, ptr, adj, head, cost, time, zm, nzm):
    n = ptr.size - 1
    dist = np.full(n, np.inf)
    tt = np.zeros(n)
    pred = np.full(n, -1)
    order = np.empty(n, np.int64)
    seen = np.zeros(n, np.bool_)
    hd = np.empty(adj.size + 1)
    hn = np.empty(adj.size + 1, np.int64)
    hd[0], hn[0], s, k = 0.0, src, 1, 0
    dist[src] = 0.0
    while s:
        d, u = hd[0], hn[0]
        s -= 1
        hd[0], hn[0] = hd[s], hn[s]
        i = 0
        while True:
            m = 2 * i + 1
            if m >= s:
                break
            if m + 1 < s and hd[m + 1] < hd[m]:
                m += 1
            if hd[i] <= hd[m]:
                break
            hd[i], hd[m] = hd[m], hd[i]
            hn[i], hn[m] = hn[m], hn[i]
            i = m
        if seen[u]:
            continue
        seen[u] = True
        order[k] = u
        k += 1
        nzm -= zm[u]
        if not nzm:
            break
        for e in adj[ptr[u]:ptr[u + 1]]:
            w = head[e]
            nd = d + cost[e]
            if nd < dist[w]:
                dist[w] = nd
                tt[w] = tt[u] + time[e]
                pred[w] = e
                i = s
                hd[i], hn[i] = nd, w
                s += 1
                while i:
                    m = (i - 1) >> 1
                    if hd[m] <= hd[i]:
                        break
                    hd[i], hd[m] = hd[m], hd[i]
                    hn[i], hn[m] = hn[m], hn[i]
                    i = m
    return dist, tt, pred, order, k


@njit(parallel=True, cache=True)
def _aon(ptr, adj, tail, head, cost, time, znode, T, K, qt, theta, occ, pcu):
    nz, nl, n = znode.size, tail.size, ptr.size - 1
    zm = np.zeros(n, np.int64)
    zm[znode] = 1
    nzm = zm.sum()
    fv = np.zeros((32, nl))
    ft = np.zeros((32, nl))
    qs = np.zeros((nz, nz))
    gt = np.empty((nz, nz))
    for c in prange(32):
        for o in range(c, nz, 32):
            dist, tt, pred, order, k = _tree(znode[o], ptr, adj, head, cost, time, zm, nzm)
            dv = np.zeros(n)
            dt = np.zeros(n)
            for d in range(nz):
                j = znode[d]
                gt[o, d] = tt[j]
                if dist[j] < np.inf:
                    qs[o, d] = T[o, d] / (1.0 + np.exp(min(theta * (dist[j] + K[o, d]), 700.0)))
                    dv[j] += qs[o, d] / occ + pcu * qt[o, d]
                    dt[j] += qt[o, d]
            for i in range(k - 1, 0, -1):
                j = order[i]
                if dv[j] > 0.0 or dt[j] > 0.0:
                    e = pred[j]
                    u = tail[e]
                    fv[c, e] += dv[j]
                    ft[c, e] += dt[j]
                    dv[u] += dv[j]
                    dt[u] += dt[j]
    return fv.sum(0), ft.sum(0), qs, gt


def graph(tail, head, n, znode):
    hd = head.tolist()
    out, inn = [[] for _ in range(n)], [[] for _ in range(n)]
    for e, (u, w) in enumerate(zip(tail.tolist(), hd)):
        out[u].append(e)
        inn[w].append(u)
    keep = np.ones(n, bool)
    for u in range(n):
        o, i = {hd[e] for e in out[u]}, set(inn[u])
        keep[u] = not (len(o) == len(out[u]) and len(i) == len(inn[u]) and (len(o) == len(i) == 1 and o != i or len(o) == 2 and o == i))
    keep[znode] = True
    rows, cols, st, sh = [], [], [], []
    for u in np.flatnonzero(keep).tolist():
        for e in out[u]:
            prev, w = u, hd[e]
            rows.append(len(st))
            cols.append(e)
            while not keep[w]:
                e = next(f for f in out[w] if hd[f] != prev)
                rows.append(len(st))
                cols.append(e)
                prev, w = w, hd[e]
            st.append(u)
            sh.append(w)
    ix = np.cumsum(keep) - 1
    st, sh = ix[st], ix[sh]
    o = np.argsort(st, kind="stable")
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), (len(st), tail.size))
    return dict(ptr=np.searchsorted(st[o], np.arange(keep.sum() + 1)), adj=o, tail=st, head=sh, znode=ix[znode], M=M, MT=M.T.tocsr())


def paths(g, cost, time, T, K, qt):
    fv, ft, qs, gt = _aon(g["ptr"], g["adj"], g["tail"], g["head"], g["M"] @ cost, g["M"] @ time, g["znode"], T, K, qt,
                          P.THETA, P.OCC, P.PCU)
    return g["MT"] @ fv, g["MT"] @ ft, qs, gt


@njit(cache=True)
def _slope(lam, v, dv, t0, cap, toll, q, dq, T, K, occ, theta, a, b):
    s = 0.0
    for e in range(v.size):
        s += occ * (t0[e] * (1 + a * ((v[e] + lam * dv[e]) / cap[e]) ** b) + toll[e]) * dv[e]
    for o in range(T.shape[0]):
        for d in range(T.shape[1]):
            if T[o, d] > 0:
                qq = q[o, d] + lam * dq[o, d]
                s += dq[o, d] * (K[o, d] + np.log(max(qq, 1e-12) / max(T[o, d] - qq, 1e-12)) / theta)
    return s


def equilibrium(g, t0, cap, toll, T, kc, pt, qt, v0, gap):
    K = kc - pt

    def times(v):
        return t0 * (1 + P.BPR_A * (v / cap) ** P.BPR_B)

    def slope(lam, dv, dq):
        return _slope(lam, v, dv, t0, cap, toll, q, dq, T, K, P.OCC, P.THETA, P.BPR_A, P.BPR_B)

    v, q, s = v0, T / 2, None
    for it in range(P.MAXIT):
        t = times(v)
        y, yt, qs, _ = paths(g, t + toll, t, T, K, qt)
        if not it:
            v, vt, q = y, yt, qs
            continue
        rel = -slope(0.0, y - v, qs - q) / (P.OCC * ((t + toll) * v).sum() + ((T - q) * pt).sum())
        if rel < gap:
            break
        if s is not None:
            hv = t0 * (v / cap) ** (P.BPR_B - 1) / cap
            a = s[0] - v
            den = (hv * (y - s[0]) * a).sum()
            al = min(max((hv * (y - v) * a).sum() / den, 0.0), 0.99) if den else 0.0
            s = (al * s[0] + (1 - al) * y, al * s[1] + (1 - al) * qs, al * s[2] + (1 - al) * yt)
        if s is None or slope(0.0, s[0] - v, s[1] - q) >= 0:
            s = (y, qs, yt)
        dv, dq = s[0] - v, s[1] - q
        lo, hi = (1.0, 1.0) if slope(1.0, dv, dq) <= 0 else (0.0, 1.0)
        while hi - lo > P.LINE:
            m = (lo + hi) / 2
            lo, hi = (m, hi) if slope(m, dv, dq) < 0 else (lo, m)
        lam = (lo + hi) / 2
        v, vt, q = v + lam * dv, vt + lam * (s[2] - vt), q + lam * dq
    return dict(v=v, vt=vt, t=times(v), q=q, p=T - q, it=it, gap=rel)
