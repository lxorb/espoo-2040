import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import params as P
from .assign import equilibrium, graph, remap, skim

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
LK = ("a", "b", "len", "lanes", "kmh", "cap", "seg", "osm")
NOTHING, WARM = {}, {}


class City:
    def __init__(self, seed, years=15, runout=5, data=OUT, who="agent", fast=False):
        self.seed, self.years, self.runout, self.data, self.who = seed, years, runout, Path(data), who
        self.gap = P.FAST if fast else P.GAP
        net = json.loads((self.data / "net.json").read_text())
        zj = json.loads((self.data / "zones.json").read_text())
        cat = self.data.parent / "catalogue.json"
        self.cat = {p["id"]: p for p in (json.loads(cat.read_text()) if cat.exists() else [])}
        self.menu = list(self.cat)
        self.nodes = np.array(net["nodes"], float)
        self.recs = []
        s = self.s = dict(year=0, cash=P.INCOME, due={}, prog={}, built={}, car=None, segname={g["id"]: g["name"] for g in net["segs"]})
        self._add(net["links"])
        s["nseg"] = int(s["seg"].max()) + 1
        segs = np.unique(s["seg"])
        s["cond"] = np.where(s["live"], np.random.default_rng(P.WORLD).uniform(*P.COND0, segs.size)[np.searchsorted(segs, s["seg"])], 1.0)
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        zs = zj["zones"]
        self.zxy = np.array([[z["x"], z["y"]] for z in zs], float)
        self.pop0, self.jobs0 = (np.array([z[k] for z in zs], float) for k in ("pop", "jobs"))
        ix = {z["id"]: i for i, z in enumerate(zs)}
        gr = zj["growth"]
        self.areas = [(a["name"], [ix[z] for z in a["zones"]]) for a in gr.get("areas", [])]
        self.rp, self.rj, self.mp, self.mj = (np.full(len(zs), float(gr[k])) for k in ("pop", "jobs", "pop", "jobs"))
        boom = rng.choice(len(self.areas), rng.integers(P.BOOMS[0], P.BOOMS[1] + 1), replace=False)
        for i, ar in enumerate(gr.get("areas", [])):
            m = self.areas[i][1]
            for r, med, k in ((self.rp, self.mp, "pop"), (self.rj, self.mj, "jobs")):
                r[m] = rng.uniform(*ar[k]) if i in boom else r[m]
                med[m] += np.mean(P.BOOMS) / len(self.areas) * np.mean(ar[k])
        n, L = len(self.nodes), s["L"]
        _, lab = connected_components(csr_matrix((np.ones(L.size), (s["a"][L], s["b"][L])), (n, n)), connection="strong")
        big = lab == np.bincount(lab).argmax()
        fast = np.ones(n, bool)
        fast[np.r_[s["a"][L], s["b"][L]].astype(int)] = False
        k = L[np.isin(s["cls"][L], ["motorway", "trunk", "ramp"])]
        fast[np.r_[s["a"][k], s["b"][k]].astype(int)] = True
        self.ok = np.flatnonzero(big)
        gates = np.array([g["node"] for g in net["gates"]])
        ext = np.array([z["ext"] for z in zs])
        self.znode = np.zeros(len(zs), np.int64)
        for m, c in ((ext, gates[big[gates]]), (~ext, np.flatnonzero(big & ~fast))):
            self.znode[m] = c[cKDTree(self.nodes[c]).query(self.zxy[m])[1]]
        same = self.znode[:, None] == self.znode[None, :]
        self.od0, self.truck0 = (np.where(same, 0.0, np.load(self.data / f)) for f in ("od.npy", "truck.npy"))
        s["fp"], s["fj"] = np.ones(len(zs)), np.ones(len(zs))
        self.ff0 = skim(self._graph(), self._t0()[L])
        xy = self.nodes[s["a"].astype(int)]
        self.foot = {k: np.isin(s["osm"], [o for e in p.get("mod", []) + p.get("works", []) for o in e["osm"]])
                     | (np.linalg.norm(xy[:, None] - np.array([e["geom"][i] for e in p.get("new", []) for i in (0, -1)] or [[1e9, 1e9]])[None], axis=2) < 300).any(1)
                     for k, p in self.cat.items()}
        self.share = 1 / (1 + np.exp(P.THETA * (self.ff0 + P.ASC_CAR + P.TERMINAL - np.load(self.data / "pt.npy")[0])))
        if str(self.data) not in WARM:
            f = self.fork(False)
            f.step()
            WARM[str(self.data)] = f.s["eq"]
        s["eq"] = copy.deepcopy(WARM[str(self.data)])

    def _add(self, links):
        s = self.s
        col = {k: np.array([-1 if l.get(k) is None else l[k] for l in links], float) for k in LK}
        cls = np.array([l["cls"] for l in links])
        col["lanes"] = np.maximum(col["lanes"], 1)
        col["cap"] *= np.array([P.CAPF[c] for c in cls])
        live = ~np.isin(cls, P.DROP) if "live" not in s else np.ones(len(links), bool)
        col.update(cap0=col["cap"], cond=np.ones(len(links)), wear=np.array([P.WEAR[c] for c in cls]) * live, life=np.array([P.LIFE[c] for c in cls]),
                   cls=cls, live=live,
                   **{k: np.zeros(len(links)) for k in ("heavy", "carv")})
        for k, v in col.items():
            s[k] = np.concatenate([s[k], v]) if k in s else v
        s["L"], s["g"] = np.flatnonzero(s["live"]), None

    def _graph(self):
        s = self.s
        if s["g"] is None:
            old, s["g"] = s.get("g0"), graph(s["a"][s["L"]].astype(np.int64), s["b"][s["L"]].astype(np.int64), len(self.nodes), self.znode)
            s["eq"] = s.get("eq") and remap(s["eq"], old, s["g"])
            s["g0"] = s["g"]
        return s["g"]

    def _t0(self):
        s = self.s
        return s["len"] / 1000 / s["kmh"] * 60 / (1 - P.SPEED_LOSS * (1 - s["cond"]))

    def _opex(self):
        return sum(self.cat[p].get("opex", 0) for p in self.s["built"])

    def _afford(self, cost, n):
        s, c = self.s, self.s["cash"]
        for k in range(n):
            c += (P.INCOME - s["due"].get(s["year"] + k, 0) - self._opex()) if k else 0
            c -= cost / n
            if c < -1e-9:
                return False
        return True

    def _act(self, a, ev):
        s = self.s
        if set(a) == {"maintain"}:
            if a["maintain"] not in range(len(P.MAINT)):
                return "bad maintenance level"
            s["maint"] = a["maintain"]
            return None
        p = self.cat.get(a.get("build")) if set(a) == {"build"} else None
        n = max(p["years"], 1) if p else 1
        why = ("unknown action" if p is None else "already started" if a["build"] in s["prog"] or a["build"] in s["built"]
               else "too many builds" if s["nb"] >= P.BUILDS else "over budget" if not self._afford(p["cost"], n) else None)
        if not why:
            s["nb"] += 1
            s["cash"] -= p["cost"] / n
            for k in range(1, n):
                s["due"][s["year"] + k] = s["due"].get(s["year"] + k, 0) + p["cost"] / n
            s["prog"][a["build"]] = p["years"]
            ev.append(f"{p['name']} started")
        return why

    def _open(self, p):
        s, new = self.s, []
        tree = cKDTree(self.nodes[self.ok])
        for e in p.get("new", []):
            g = np.array(e["geom"], float)
            a, b = self.ok[tree.query(g[[0, -1]])[1]]
            ln = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
            s["segname"][s["nseg"]] = p["name"]
            for u, w, gg in [(a, b, g)] + ([] if e.get("oneway") else [(b, a, g[::-1])]):
                new.append(dict(a=int(u), b=int(w), len=ln, lanes=e["lanes"], kmh=e["kmh"], cap=P.CAP_LANE[e["cls"]] * e["lanes"],
                                cls=e["cls"], name=p["name"], ref="", osm=None, seg=s["nseg"], bridge=e.get("bridge", False), geom=gg.tolist()))
            s["nseg"] += 1
        if new:
            self._add(new)
        for m in p.get("mod", []):
            k = np.isin(s["osm"], m["osm"])
            nl = np.maximum(s["lanes"][k] + m.get("lanes", 0), 1)
            s["cap"][k] *= nl / s["lanes"][k] * m.get("cap", 1.0)
            s["lanes"][k] = nl
            s["kmh"][k] = m.get("kmh", s["kmh"][k])
        return new

    def _demand(self):
        s, m = self.s, self.od0.copy()
        R, C = self.od0.sum(1) * s["fp"], self.od0.sum(0) * s["fj"]
        C *= R.sum() / C.sum()
        for _ in range(10):
            m *= (R / np.maximum(m.sum(1), 1e-9))[:, None]
            m *= C / np.maximum(m.sum(0), 1e-9)
        return m, self.truck0 * np.sqrt(s["fj"][:, None] * s["fj"][None])

    def _segs(self):
        s = self.s
        ids, inv = np.unique(s["seg"], return_inverse=True)
        cond = np.full(ids.size, 1.0)
        np.minimum.at(cond, inv, s["cond"])
        return ids, inv, cond, np.bincount(inv, s["len"] * s["lanes"]) / 1000 * P.RESURF

    @property
    def done(self):
        return self.s["year"] >= self.years + self.runout

    def step(self, actions=()):
        keep = copy.deepcopy(self.s)
        try:
            return self._step(actions)
        except BaseException:
            self.s = keep
            raise

    def _step(self, actions):
        s, ev, acts = self.s, [], []
        y, c0 = s["year"], s["cash"]
        if y:
            s["fp"] *= 1 + self.rp
            s["fj"] *= 1 + self.rj
        T, qt = self._demand()
        s["nb"], s["maint"] = 0, P.MAINT_DEFAULT if y < self.years else 0
        for a in actions:
            why = (self._act(a, ev) if isinstance(a, dict) else "unknown action") if y < self.years else "runout"
            acts.append(dict(a, ok=not why, **({"why": why} if why else {})))
        new, cut, works, opened = [], np.ones(s["a"].size), [], []
        for pid in list(s["prog"]):
            p = self.cat[pid]
            if s["prog"][pid]:
                for w in p.get("works", []):
                    cut[np.isin(s["osm"][:cut.size], w["osm"])] *= w["cap"]
                s["prog"][pid] -= 1
                works.append(pid)
            else:
                s["built"][pid] = s["prog"].pop(pid) + y
                new += self._open(p)
                ev.append(f"{p['name']} opened")
                opened.append(pid)
        cut = np.r_[cut, np.ones(s["a"].size - cut.size)]
        esal = (s["heavy"] * P.TRUCK_ESAL + s["carv"] * P.CAR_ESAL) / s["lanes"]
        s["cond"] = np.clip(s["cond"] - s["wear"] - esal / s["life"], 0, 1)
        ids, inv, cond, cost = self._segs()
        left = min(P.MAINT[s["maint"]], max(s["cash"], 0.0))
        fix = []
        for g in np.argsort(cond):
            if cond[g] < P.MAINT_BELOW and cost[g] <= left:
                fix.append(g)
                left -= cost[g]
        s["cash"] -= min(P.MAINT[s["maint"]], max(s["cash"], 0.0)) - left
        bad = (cond < P.COND_MIN) & ~np.isin(np.arange(ids.size), fix)
        s["cash"] -= P.REBUILD * cost[bad].sum()
        ev += [f"{s['segname'].get(int(g), g)} emergency rebuild" for g in ids[bad]]
        cut[bad[inv]] *= P.REBUILD_CAP
        s["cond"][np.isin(inv, fix) | bad[inv]] = 1.0
        L, t0 = s["L"], self._t0()
        r = equilibrium(self._graph(), t0[L], (s["cap"] * cut)[L], 0.0, T * self.share, qt, s.get("eq"), self.gap)
        s["eq"] = r["st"]
        s["heavy"], s["carv"], t = np.zeros(s["a"].size), np.zeros(s["a"].size), t0.copy()
        s["heavy"][L], s["carv"][L], t[L] = r["vt"], r["v"] - P.PCU * r["vt"], r["t"]
        s["car"] = float((T * self.share).sum() / T.sum())
        delay = jam = float(((s["carv"] * P.OCC + s["heavy"]) * (t - s["len"] / 1000 / s["kmh"] * 60)).sum() / 60 * P.ANNUAL)
        rec = dict(year=P.Y0 + y, budget=s["cash"], spent=c0 - s["cash"], actions=acts, maint=s["maint"], delay=delay, jam=jam,
                   car=s["car"], flow=s["carv"] + s["heavy"], time=t * 60, cond=s["cond"].copy(), cap=s["cap"] * cut / s["cap0"], new=new,
                   works=works, open=opened, events=ev, it=r["it"], gap=float(r["gap"]))
        s["year"] += 1
        s["cash"] += P.INCOME - s["due"].pop(s["year"], 0.0) - self._opex()
        self.recs.append(rec)
        return rec

    def state(self):
        s, r = self.s, self.recs[-1] if self.recs else {}
        pop, jobs = self.pop0 * s["fp"], self.jobs0 * s["fj"]
        cond = self._segs()[2]
        return dict(year=P.Y0 + s["year"], step=s["year"], budget=s["cash"], income=P.INCOME, opex=self._opex(),
                    committed=sum(s["due"].values()), car=s["car"], last=r.get("delay"), jam=r.get("jam"),
                    delay=sum(r["delay"] for r in self.recs), pop=pop.tolist(), jobs=jobs.tolist(),
                    areas=[dict(name=n, pop=float(pop[m].sum()), jobs=float(jobs[m].sum()), pop0=float(self.pop0[m].sum()),
                                jobs0=float(self.jobs0[m].sum())) for n, m in self.areas],
                    works=dict(s["prog"]), built=list(s["built"]), cond=[float(cond.mean()), float(np.sort(cond)[:max(cond.size // 10, 1)].mean())],
                    done=self.done)

    def catalogue(self):
        s, r = self.s, self.recs[-1] if self.recs else None
        n = self.foot[self.menu[0]].size
        late = np.maximum(r["flow"][:n] * (r["time"][:n] - s["len"][:n] / s["kmh"][:n] * 3600 / 1000), 0) if r else np.zeros(n)
        return [dict(id=k, name=p["name"], kind=p["kind"], cost=p["cost"], years=p["years"], opex=p.get("opex", 0),
                     afford=self._afford(p["cost"], max(p["years"], 1)), jam=round(float(late[self.foot[k]].sum() / max(late.sum(), 1e-9)), 3))
                for k, p in self.cat.items() if k not in s["prog"] and k not in s["built"]]

    def mask(self):
        s = self.s
        return [s["year"] < self.years and k not in s["prog"] and k not in s["built"] and self._afford(p["cost"], max(p["years"], 1))
                for k, p in self.cat.items()]

    def fork(self, median=True):
        f = copy.copy(self)
        f.s, f.recs = copy.deepcopy(self.s), list(self.recs)
        if median:
            w = self.s["year"] / (self.s["year"] + P.LEARN)
            f.rp, f.rj, f.who = self.mp + w * (self.rp - self.mp), self.mj + w * (self.rj - self.mj), "forecast"
        return f

    def check(self, actions):
        f = self.fork(False)
        f.s["nb"] = 0
        return [f._act(a, []) if f.s["year"] < self.years else "runout" for a in actions]

    def forecast(self, plan, years):
        f = self.fork()
        return [f.step(plan.get(k, [])) for k in range(min(years, self.years + self.runout - self.s["year"]))]

    def baseline(self):
        files = [self.data / f for f in ("net.json", "zones.json", "od.npy", "truck.npy", "pt.npy")] + sorted(Path(__file__).parent.glob("*.py"))
        key = hashlib.sha1(repr([(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files] + [self.seed, self.years, self.runout]).encode()).hexdigest()
        path = ROOT / "runs" / ".nothing.json"
        if key not in NOTHING:
            NOTHING.update(json.loads(path.read_text()) if path.exists() else {})
        if key not in NOTHING:
            c = City(self.seed, self.years, self.runout, self.data, "nothing")
            while not c.done:
                c.step()
            NOTHING[key] = [r["delay"] for r in c.recs]
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}")
            tmp.write_text(json.dumps({**(json.loads(path.read_text()) if path.exists() else {}), key: NOTHING[key]}))
            os.replace(tmp, path)
        return NOTHING[key]

    def result(self):
        if not self.done:
            return None
        norms = Path(__file__).parent / "norms.json"
        base, best = (json.loads(norms.read_text()) if norms.exists() else {}).get(str(self.seed) if (self.years, self.runout) == (15, 5) else "", (None, None))
        run, base = sum(r["delay"] for r in self.recs), base or sum(self.baseline())
        return dict(delay=run, nothing=base, saving=base - run, score=None if best is None or base == best else (base - run) / (base - best))

    def score(self):
        r = self.result()
        return r and r["score"]

    def log(self, name=None):
        nd = dict(flow=0, time=1, cond=2, cap=2)

        def js(k, v):
            return v if not isinstance(v, np.ndarray) else np.round(v).astype(int).tolist() if not nd.get(k, 3) else np.round(v, nd.get(k, 3)).tolist()

        d = dict(seed=self.seed, who=self.who, y0=P.Y0, score=self.score(), years=[{k: js(k, v) for k, v in r.items()} for r in self.recs])
        if name:
            (ROOT / "runs").mkdir(exist_ok=True)
            (ROOT / "runs" / f"{name}.json").write_text(json.dumps(d))
        return d
