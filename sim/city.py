import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import params as P
from .assign import equilibrium, graph, paths

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
LK = ("a", "b", "len", "lanes", "kmh", "cap", "seg", "osm")
NOTHING = {}


class City:
    def __init__(self, seed, years=15, runout=5, data=OUT, who="agent", fast=P.FAST):
        self.seed, self.years, self.runout, self.data, self.who = seed, years, runout, Path(data), who
        self.gap, self.fast = P.GAP, fast
        net = json.loads((self.data / "net.json").read_text())
        zj = json.loads((self.data / "zones.json").read_text())
        cat = self.data.parent / "catalogue.json"
        self.cat = {p["id"]: p for p in (json.loads(cat.read_text()) if cat.exists() else [])}
        self.nodes = np.array(net["nodes"], float)
        self.recs = []
        s = self.s = dict(year=0, cash=P.INCOME, due={}, prog={}, built={}, resurf=set(), car=None,
                          segname={g["id"]: g["name"] for g in net["segs"]})
        self._add(net["links"])
        s["nseg"] = int(s["seg"].max()) + 1
        segs = np.unique(s["seg"])
        s["cond"] = np.random.default_rng(P.WORLD).uniform(*P.COND0, segs.size)[np.searchsorted(segs, s["seg"])]
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        zs = zj["zones"]
        self.zxy = np.array([[z["x"], z["y"]] for z in zs], float)
        self.pop0, self.jobs0 = (np.array([z[k] for z in zs], float) for k in ("pop", "jobs"))
        ix = {z["id"]: i for i, z in enumerate(zs)}
        gr = zj["growth"]
        self.rp, self.rj, self.mp, self.mj = (np.full(len(zs), float(gr[k])) for k in ("pop", "jobs", "pop", "jobs"))
        for ar in gr.get("areas", []):
            m = [ix[z] for z in ar["zones"]]
            for r, med, k in ((self.rp, self.mp, "pop"), (self.rj, self.mj, "jobs")):
                r[m] = rng.uniform(*ar[k])
                med[m] = sum(ar[k]) / 2
        n = len(self.nodes)
        _, lab = connected_components(csr_matrix((np.ones(s["a"].size), (s["a"], s["b"])), (n, n)), connection="strong")
        big = lab == np.bincount(lab).argmax()
        fast = np.zeros(n, bool)
        k = np.isin(s["cls"], ["motorway", "trunk", "ramp"])
        fast[np.r_[s["a"][k], s["b"][k]].astype(int)] = True
        self.ok = np.flatnonzero(big)
        gates = np.array([g["node"] for g in net["gates"]])
        ext = np.array([z["ext"] for z in zs])
        self.znode = np.zeros(len(zs), np.int64)
        for m, c in ((ext, gates[big[gates]]), (~ext, np.flatnonzero(big & ~fast))):
            self.znode[m] = c[cKDTree(self.nodes[c]).query(self.zxy[m])[1]]
        same = self.znode[:, None] == self.znode[None, :]
        self.od0, self.truck0 = (np.where(same, 0.0, np.load(self.data / f)) for f in ("od.npy", "truck.npy"))
        s["pt"] = np.load(self.data / "pt.npy").astype(float)
        s["park"] = np.zeros(len(zs))
        s["fp"], s["fj"] = np.ones(len(zs)), np.ones(len(zs))
        t0 = self._t0()
        z = np.zeros_like(self.od0)
        self.ff0 = paths(self._graph(), t0, t0, z, z, z)[3]

    def _add(self, links):
        s = self.s
        col = {k: np.array([-1 if l.get(k) is None else l[k] for l in links], float) for k in LK}
        cls = np.array([l["cls"] for l in links])
        col["lanes"] = np.maximum(col["lanes"], 1)
        col["cap"] *= np.array([P.CAPF[c] for c in cls])
        col.update(cap0=col["cap"], cond=np.ones(len(links)), wear=np.array([P.WEAR[c] for c in cls]),
                   life=np.array([P.LIFE[c] for c in cls]), cls=cls,
                   **{k: np.zeros(len(links)) for k in ("retime", "toll", "heavy", "carv")})
        for k, v in col.items():
            s[k] = np.concatenate([s[k], v]) if k in s else v
        s["g"] = None

    def _graph(self):
        s = self.s
        if s["g"] is None:
            s["g"] = graph(s["a"].astype(np.int64), s["b"].astype(np.int64), len(self.nodes), self.znode)
        return s["g"]

    def _t0(self):
        s = self.s
        return s["len"] / 1000 / s["kmh"] * 60 / (1 - P.SPEED_LOSS * (1 - s["cond"]))

    def _opex(self):
        return sum(self.cat[p].get("opex", 0) for p in self.s["built"])

    def _lanekm(self, m):
        return (self.s["len"][m] * self.s["lanes"][m]).sum() / 1000

    def _afford(self, cost, n):
        s, c = self.s, self.s["cash"]
        for k in range(n):
            c += (P.INCOME - s["due"].get(s["year"] + k, 0) - self._opex()) if k else 0
            c -= cost / n
            if c < -1e-9:
                return False
        return True

    def _pay(self, cost, n):
        s = self.s
        s["cash"] -= cost / n
        for k in range(1, n):
            s["due"][s["year"] + k] = s["due"].get(s["year"] + k, 0) + cost / n

    def _act(self, a, ev):
        s = self.s
        if "build" in a:
            p = self.cat.get(a["build"])
            lv = a.get("level", 0)
            n = max(p["years"], 1) if p else 1
            why = ("unknown project" if p is None else "already started" if a["build"] in s["prog"] or a["build"] in s["built"]
                   else "bad level" if not 0 <= lv < len((p.get("toll") or p.get("park") or {"eur": [0]})["eur"])
                   else "over budget" if not self._afford(p["cost"], n) else None)
            if not why:
                self._pay(p["cost"], n)
                s["prog"][a["build"]] = [p["years"], lv]
                ev.append(f"{p['name']} started")
            return why
        seg = a.get("resurface", a.get("retime"))
        m = s["seg"] == seg
        cost = P.RESURF * self._lanekm(m) if "resurface" in a else P.RETIME_COST
        why = ("unknown action" if seg is None else "unknown seg" if not m.any()
               else "already resurfaced" if "resurface" in a and seg in s["resurf"]
               else "over budget" if not self._afford(cost, 1) else None)
        if not why:
            self._pay(cost, 1)
            if "resurface" in a:
                s["resurf"].add(seg)
            else:
                s["retime"][m] = P.RETIME_GAIN
        return why

    def _open(self, p, lv):
        s, new = self.s, []
        tree = cKDTree(self.nodes[self.ok])
        for e in p.get("new", []):
            g = np.array(e["geom"], float)
            a, b = self.ok[tree.query(g[[0, -1]])[1]]
            ln = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
            s["segname"][s["nseg"]] = p["name"]
            for u, w, gg in [(a, b, g)] + ([] if e.get("oneway") else [(b, a, g[::-1])]):
                new.append(dict(a=int(u), b=int(w), len=ln, lanes=e["lanes"], kmh=e["kmh"], cap=P.CAP_LANE[e["cls"]] * e["lanes"],
                                cls=e["cls"], name=p["name"], ref="", osm=None, seg=s["nseg"],
                                bridge=e.get("bridge", False), geom=gg.tolist()))
            s["nseg"] += 1
        if new:
            self._add(new)
        for m in p.get("mod", []):
            k = np.isin(s["osm"], m["osm"])
            nl = np.maximum(s["lanes"][k] + m.get("lanes", 0), 1)
            s["cap"][k] *= nl / s["lanes"][k] * m.get("cap", 1.0)
            s["lanes"][k] = nl
            s["kmh"][k] = m.get("kmh", s["kmh"][k])
        for ln in p.get("pt", []):
            st = np.array(ln["stops"], float)
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(st, axis=0), axis=1))]
            d = np.linalg.norm(self.zxy[:, None] - st[None], axis=2)
            i, w = d.argmin(1), d.min(1)
            t = ((w[:, None] + w[None]) / 1000 / P.WALK_KMH * 60 + ln["headway"] / 2
                 + np.abs(cum[i][:, None] - cum[i][None]) / 1000 / ln["kmh"] * 60)
            near = (w[:, None] <= P.STOP_R) & (w[None] <= P.STOP_R)
            s["pt"] = np.where(near, np.minimum(s["pt"], t), s["pt"])
        if "pnr" in p:
            k = cKDTree(self.zxy).query(p["pnr"]["at"])[1]
            alt = self.ff0[:, k:k + 1] + s["pt"][k] + P.PNR_PEN
            gain = (self.ff0[:, k] <= P.PNR_REACH)[:, None] & (alt < s["pt"])
            w = min(1.0, p["pnr"]["spaces"] / max((self.od0 * gain).sum(), 1e-9))
            s["pt"] = np.where(gain, s["pt"] - w * (s["pt"] - alt), s["pt"])
        if "toll" in p:
            s["toll"][np.isin(s["osm"], p["toll"]["osm"])] = p["toll"]["eur"][lv]
        if "park" in p:
            s["park"][shapely.contains_xy(shapely.Polygon(p["park"]["area"]), *self.zxy.T)] = p["park"]["eur"][lv]
        return new

    def _demand(self):
        s, m = self.s, self.od0.copy()
        R, C = self.od0.sum(1) * s["fp"], self.od0.sum(0) * s["fj"]
        C *= R.sum() / C.sum()
        for _ in range(10):
            m *= (R / np.maximum(m.sum(1), 1e-9))[:, None]
            m *= C / np.maximum(m.sum(0), 1e-9)
        return m, self.truck0 * np.sqrt(s["fj"][:, None] * s["fj"][None])

    @property
    def done(self):
        return self.s["year"] >= self.years + self.runout

    def step(self, actions=()):
        s, ev, acts = self.s, [], []
        y, c0 = s["year"], s["cash"]
        if y:
            s["fp"] *= 1 + self.rp
            s["fj"] *= 1 + self.rj
        T, qt = self._demand()
        s["resurf"] = set()
        for a in actions:
            why = self._act(a, ev) if y < self.years else "runout"
            acts.append(dict(a, ok=not why, **({"why": why} if why else {})))
        new, works = [], []
        for pid in list(s["prog"]):
            p = self.cat[pid]
            if s["prog"][pid][0]:
                works += p.get("works", [])
                s["prog"][pid][0] -= 1
            else:
                s["built"][pid] = s["prog"].pop(pid)[1]
                new += self._open(p, s["built"][pid])
                ev.append(f"{p['name']} opened")
        cut = 1 + s["retime"]
        for w in works:
            cut[np.isin(s["osm"], w["osm"])] *= w["cap"]
        esal = (s["heavy"] * P.TRUCK_ESAL + s["carv"] * P.CAR_ESAL) / s["lanes"]
        s["cond"] = np.clip(s["cond"] - s["wear"] - esal / s["life"], 0, 1)
        res = np.isin(s["seg"], list(s["resurf"]))
        bad = np.isin(s["seg"], s["seg"][(s["cond"] < P.COND_MIN) & ~res])
        cut[res] *= P.RESURF_CAP
        cut[bad] *= P.REBUILD_CAP
        s["cash"] -= P.REBUILD * P.RESURF * self._lanekm(bad)
        ev += [f"{s['segname'].get(int(g), g)} emergency rebuild" for g in np.unique(s["seg"][bad])]
        s["cond"][res | bad] = 1.0
        kc = np.broadcast_to(P.ASC_CAR + P.TERMINAL + s["park"] / P.VOT * 60, T.shape)
        f = np.clip(((s["p0"] + 1e-3) / (s["plast"] + 1e-3)) ** 0.5, *P.MOHRING) if y else 1.0
        pt = s["pt"] * (1 + P.WAIT * (f - 1))
        r = equilibrium(self._graph(), self._t0(), s["cap"] * cut, s["toll"] / P.VOT * 60, T, kc, pt, qt,
                        s["carv"] + P.PCU * s["heavy"], self.gap)
        s["plast"] = r["p"]
        s.setdefault("p0", r["p"])
        s["heavy"], s["carv"] = r["vt"], r["v"] - P.PCU * r["vt"]
        s["retime"] *= P.RETIME_KEEP
        s["car"] = float(r["q"].sum() / T.sum())
        delay = (P.OCC * (s["carv"] * r["t"]).sum() - (r["q"] * (self.ff0 - P.TERMINAL)).sum() + (r["p"] * (pt - self.ff0)).sum()) / 60 * P.ANNUAL
        rec = dict(year=P.Y0 + y, budget=s["cash"], spent=c0 - s["cash"], actions=acts, delay=float(delay), car=s["car"],
                   flow=s["carv"] + r["vt"], time=r["t"] * 60, cond=s["cond"].copy(), cap=s["cap"] * cut / s["cap0"], new=new,
                   events=ev, it=r["it"], gap=float(r["gap"]))
        s["year"] += 1
        s["cash"] += P.INCOME - s["due"].pop(s["year"], 0.0) - self._opex()
        self.recs.append(rec)
        return rec

    def state(self):
        s = self.s
        ids, inv = np.unique(s["seg"], return_inverse=True)
        cond = np.full(ids.size, 1.0)
        np.minimum.at(cond, inv, s["cond"])
        lk = np.bincount(inv, s["len"] * s["lanes"]) / 1000
        return dict(year=P.Y0 + s["year"], step=s["year"], budget=s["cash"], income=P.INCOME, opex=self._opex(),
                    committed=sum(s["due"].values()), car=s["car"], pop=(self.pop0 * s["fp"]).tolist(),
                    jobs=(self.jobs0 * s["fj"]).tolist(), works={k: v[0] for k, v in s["prog"].items()}, built=dict(s["built"]),
                    segs=[dict(id=int(g), name=s["segname"].get(int(g), ""), cond=float(c), lanekm=float(l), resurface=P.RESURF * l)
                          for g, c, l in zip(ids, cond, lk)],
                    delay=sum(r["delay"] for r in self.recs), done=self.done)

    def catalogue(self):
        s = self.s
        return [dict(id=k, name=p["name"], kind=p["kind"], cost=p["cost"], years=p["years"], opex=p.get("opex", 0),
                     levels=(p.get("toll") or p.get("park") or {}).get("eur"), afford=self._afford(p["cost"], max(p["years"], 1)))
                for k, p in self.cat.items() if k not in s["prog"] and k not in s["built"]]

    def forecast(self, plan, years):
        f = copy.copy(self)
        f.s, f.recs, f.rp, f.rj, f.who, f.gap = copy.deepcopy(self.s), [], self.mp, self.mj, "forecast", self.fast
        return [f.step(plan.get(k, [])) for k in range(min(years, self.years + self.runout - self.s["year"]))]

    def _nothing(self):
        files = [self.data / f for f in ("net.json", "zones.json", "od.npy", "truck.npy", "pt.npy")] + sorted(Path(__file__).parent.glob("*.py"))
        key = hashlib.sha1(repr([(str(f), f.stat().st_mtime_ns) for f in files] + [self.seed, self.years, self.runout]).encode()).hexdigest()
        if key not in NOTHING:
            path = ROOT / "runs" / ".nothing.json"
            disk = json.loads(path.read_text()) if path.exists() else {}
            if key not in disk:
                c = City(self.seed, self.years, self.runout, self.data, "nothing")
                while not c.done:
                    c.step()
                disk[key] = sum(r["delay"] for r in c.recs)
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(disk))
            NOTHING[key] = disk[key]
        return NOTHING[key]

    def result(self):
        if not self.done:
            return None
        norms = Path(__file__).parent / "norms.json"
        best = json.loads(norms.read_text()).get(str(self.seed)) if norms.exists() else None
        run, base = sum(r["delay"] for r in self.recs), self._nothing()
        return dict(delay=run, nothing=base, saving=base - run,
                    score=None if best is None or base == best else (base - run) / (base - best))

    def score(self):
        r = self.result()
        return r and r["score"]

    def log(self, name=None):
        nd = dict(flow=0, time=1, cond=2, cap=2)

        def js(k, v):
            return v if not isinstance(v, np.ndarray) else np.round(v).astype(int).tolist() if not nd.get(k, 3) else np.round(v, nd.get(k, 3)).tolist()

        d = dict(seed=self.seed, who=self.who, y0=P.Y0, score=self.score(),
                 years=[{k: js(k, v) for k, v in r.items()} for r in self.recs])
        if name:
            (ROOT / "runs").mkdir(exist_ok=True)
            (ROOT / "runs" / f"{name}.json").write_text(json.dumps(d))
        return d
