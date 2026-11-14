import numpy as np
from numba import njit
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


@njit(cache=True)
def _move(pool, a, n, x, xt, v, vt, cost, der, c0, c1, pw):
    for i in range(a, a + n):
        e = pool[i]
        v[e] = max(v[e] + x, 0.0)
        vt[e] += xt
        cost[e] = c0[e] + c1[e] * v[e] ** pw
        der[e] = pw * c1[e] * v[e] ** (pw - 1)


@njit(cache=True, error_model="numpy")
def _sweep(ptr, adj, tail, head, znode, zm, cost, der, c0, c1, pw, T, qt, occ, pcu, pool, top, pst, pln, h, ht, v, vt):
    Z, MP = T.shape[0], pln.shape[1]
    gn, gd, full = 0.0, 0.0, False
    C, S = np.zeros(MP), np.zeros(MP)
    mark, stamp = np.zeros(v.size, np.int64), 0
    for o in range(Z):
        dist, _, pred, _, _ = _tree(znode[o], ptr, adj, head, cost, cost, zm, zm.sum())
        for d in range(Z):
            if T[o, d] <= 0 and qt[o, d] <= 0:
                continue
            k = o * Z + d
            for j in range(MP):
                C[j], S[j] = 1e30, 0.0
                if pln[k, j]:
                    C[j], S[j] = 0.0, 0.0
                    for i in range(pst[k, j], pst[k, j] + pln[k, j]):
                        C[j] += cost[pool[i]]
                        S[j] += der[pool[i]]
            cd = dist[znode[d]]
            if cd < C.min() - 1e-9:
                n, u = 0, znode[d]
                while u != znode[o]:
                    n += 1
                    u = tail[pred[u]]
                if top + n > pool.size:
                    full = True
                else:
                    w = h[k] / occ + pcu * ht[k] - (pln[k] == 0)
                    j = w.argmin()
                    u, i = znode[d], top + n
                    while u != znode[o]:
                        i -= 1
                        pool[i] = pred[u]
                        u = tail[pred[u]]
                    if pln[k, j]:
                        f = h[k, j] / occ + pcu * ht[k, j]
                        _move(pool, pst[k, j], pln[k, j], -f, -ht[k, j], v, vt, cost, der, c0, c1, pw)
                        _move(pool, top, n, f, ht[k, j], v, vt, cost, der, c0, c1, pw)
                    pst[k, j], pln[k, j] = top, n
                    top += n
                    C[j], S[j] = 0.0, 0.0
                    for i in range(pst[k, j], pst[k, j] + n):
                        C[j] += cost[pool[i]]
                        S[j] += der[pool[i]]
            b = C.argmin()
            cm = C[b]
            y = qt[o, d] - ht[k].sum()
            if y > 1e-9:
                ht[k, b] += y
                _move(pool, pst[k, b], pln[k, b], pcu * y, y, v, vt, cost, der, c0, c1, pw)
            stamp += 1
            for i in range(pst[k, b], pst[k, b] + pln[k, b]):
                mark[pool[i]] = stamp
            hc = (h[k] * np.minimum(C, 1e12)).sum()
            gn += hc - h[k].sum() * cm
            gd += hc
            for j in range(MP):
                if j != b and pln[k, j] and C[j] > cm and (h[k, j] > 0 or ht[k, j] > 0):
                    sh = 0.0
                    for i in range(pst[k, j], pst[k, j] + pln[k, j]):
                        if mark[pool[i]] == stamp:
                            sh += der[pool[i]]
                    r = min(1.0, (C[j] - cm) / max(S[j] + S[b] - 2 * sh, 1e-12) / max(h[k, j] / occ + pcu * ht[k, j], 1e-12))
                    x, y = r * h[k, j], r * ht[k, j]
                    h[k, j] -= x
                    h[k, b] += x
                    ht[k, j] -= y
                    ht[k, b] += y
                    _move(pool, pst[k, j], pln[k, j], -x / occ - pcu * y, -y, v, vt, cost, der, c0, c1, pw)
                    _move(pool, pst[k, b], pln[k, b], x / occ + pcu * y, y, v, vt, cost, der, c0, c1, pw)
                    cm += der[pool[pst[k, b]:pst[k, b] + pln[k, b]]].sum() * (x / occ + pcu * y)
            x = T[o, d] - h[k].sum()
            if x >= 0:
                h[k, b] += x
                _move(pool, pst[k, b], pln[k, b], x / occ, 0.0, v, vt, cost, der, c0, c1, pw)
            else:
                for j in range(MP):
                    if pln[k, j] and h[k, j] > 0:
                        y = h[k, j] * x / (T[o, d] - x)
                        h[k, j] += y
                        _move(pool, pst[k, j], pln[k, j], y / occ, 0.0, v, vt, cost, der, c0, c1, pw)
    return gn, gd, top, full


@njit(cache=True)
def _pack(pool, pst, pln, h, ht, v, vt, occ, pcu, size):
    out, top = np.empty(size, pool.dtype), 0
    for k in range(pln.shape[0]):
        for j in range(pln.shape[1]):
            n = pln[k, j]
            if n and (h[k, j] > 1e-9 or ht[k, j] > 1e-9):
                out[top:top + n] = pool[pst[k, j]:pst[k, j] + n]
                pst[k, j] = top
                top += n
                for i in range(pst[k, j], top):
                    v[out[i]] += h[k, j] / occ + pcu * ht[k, j]
                    vt[out[i]] += ht[k, j]
            else:
                pln[k, j], h[k, j], ht[k, j] = 0, 0.0, 0.0
    return out, top


@njit(cache=True)
def _remap(pool, pst, pln, ptr, idx, ce, ne):
    tot = 0
    for k in range(pln.shape[0]):
        for j in range(pln.shape[1]):
            for i in range(pst[k, j], pst[k, j] + pln[k, j]):
                tot += ptr[pool[i] + 1] - ptr[pool[i]]
    out, top, mark = np.zeros(2 * tot, pool.dtype), 0, np.zeros(ne, np.int64)
    for k in range(pln.shape[0]):
        for j in range(pln.shape[1]):
            a = top
            for i in range(pst[k, j], pst[k, j] + pln[k, j]):
                for r in range(ptr[pool[i]], ptr[pool[i] + 1]):
                    e = ce[idx[r]]
                    if mark[e] != k * pln.shape[1] + j + 1:
                        mark[e] = k * pln.shape[1] + j + 1
                        out[top] = e
                        top += 1
            pst[k, j], pln[k, j] = a, top - a
    return out


def remap(st, old, new):
    ce = np.zeros(new["M"].shape[1], np.int64)
    ce[new["M"].indices] = np.repeat(np.arange(new["M"].shape[0]), np.diff(new["M"].indptr))
    st = dict(st, pst=st["pst"].copy(), pln=st["pln"].copy())
    st["pool"] = _remap(st["pool"], st["pst"], st["pln"], old["M"].indptr, old["M"].indices, ce, new["M"].shape[0])
    return st


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


def skim(g, cost):
    zm = np.zeros(g["ptr"].size - 1, np.int64)
    zm[g["znode"]] = 1
    c = g["M"] @ cost
    return np.array([_tree(o, g["ptr"], g["adj"], g["head"], c, c, zm, zm.sum())[0][g["znode"]] for o in g["znode"]])


def equilibrium(g, t0, cap, toll, T, qt, st, gap):
    Z, n, ne = T.shape[0], g["ptr"].size - 1, g["tail"].size
    zm = np.zeros(n, np.int64)
    zm[g["znode"]] = 1
    if st is None:
        st = dict(pool=np.zeros(1 << 20, np.int32), pst=np.zeros((Z * Z, P.PATHS), np.int64), pln=np.zeros((Z * Z, P.PATHS), np.int64),
                  h=np.zeros((Z * Z, P.PATHS)), ht=np.zeros((Z * Z, P.PATHS)), T=T, qt=qt)
    st = dict(st, T=T, qt=qt)
    st["h"] = st["h"] * np.divide(T, st["T"], out=np.zeros_like(T), where=st["T"] > 0).reshape(-1, 1)
    st["ht"] = st["ht"] * np.divide(qt, st["qt"], out=np.zeros_like(qt), where=st["qt"] > 0).reshape(-1, 1)
    c0, c1 = g["M"] @ (t0 + toll), g["M"] @ (t0 * P.BPR_A / cap ** P.BPR_B)
    for it in range(P.MAXIT):
        v, vt = np.zeros(ne), np.zeros(ne)
        st["pool"], top = _pack(st["pool"], st["pst"], st["pln"], st["h"], st["ht"], v, vt, P.OCC, P.PCU, st["pool"].size)
        x = g["MT"] @ v / cap
        t = t0 * (1 + P.BPR_A * x ** P.BPR_B)
        gn, gd, top, full = _sweep(g["ptr"], g["adj"], g["tail"], g["head"], g["znode"], zm, g["M"] @ (t + toll), g["M"] @ (t0 * P.BPR_A * P.BPR_B * x ** (P.BPR_B - 1) / cap), c0, c1, int(P.BPR_B),
                                   T, qt, P.OCC, P.PCU, st["pool"], top, st["pst"], st["pln"], st["h"], st["ht"], v, vt)
        if full or top > st["pool"].size // 2:
            st["pool"] = np.concatenate([st["pool"], np.zeros_like(st["pool"])])
        if it and gn < gap * gd:
            break
    vo = g["MT"] @ v
    t = t0 * (1 + P.BPR_A * (vo / cap) ** P.BPR_B)
    return dict(v=vo, vt=g["MT"] @ vt, t=t, it=it, gap=gn / max(gd, 1e-12), st=st)
