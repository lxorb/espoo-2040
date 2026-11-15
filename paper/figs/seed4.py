import datetime, json, platform, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim import City, params as P

NET = json.loads((ROOT / "data/out/net.json").read_text())["links"]
ZJ = json.loads((ROOT / "data/out/zones.json").read_text())
CAT = {p["id"]: p for p in json.loads((ROOT / "data/catalogue.json").read_text())}
ACC = [k for k in CAT if k.endswith("-access")]
L, O = "leppavaara-access", "otaniemi-access"
BASE = [[{k: v for k, v in a.items() if k != "ok"} for a in acts] for _, acts in json.loads((ROOT / "evidence/gate4-s4/plan.json").read_text())["plan"]]


def sub(plan, old, new):
    return [[dict(build=new) if a.get("build") == old else a for a in acts if new or a.get("build") != old] for acts in plan]


def add(plan, y, pid):
    return [acts + [dict(build=pid)] if i == y else acts for i, acts in enumerate(plan)]


def only(y, pid):
    return [[dict(build=pid), dict(maintain=1)] if i == y else [] for i in range(15)]


ARMS = dict(base=(BASE, {}), swap=(sub(BASE, O, L), {}), drop=(sub(BASE, O, None), {}), nothing=([], {}),
            swap_nowork=(sub(BASE, O, L), {L: "works"}), swap_nomod=(sub(BASE, O, L), {L: "mod"}), base_nowork=(BASE, {O: "works"}),
            base_nomod=(BASE, {O: "mod"}), base_lep2030=(add(BASE, 4, L), {}), base_lep2034=(add(BASE, 8, L), {}), swap_ota2034=(add(sub(BASE, O, L), 8, O), {}))


def run(seed, plan, off=None):
    c = City(seed)
    for k, f in (off or {}).items():
        c.cat[k] = dict(c.cat[k], **{f: []})
    ys = []
    while not c.done:
        y = c.s["year"]
        r = c.step(plan[y] if y < len(plan) else [])
        s, n = c.s, r["flow"].size
        d = (s["carv"] * P.OCC + s["heavy"]) * (r["time"] / 60 - s["len"] / 1000 / s["kmh"] * 60) / 60 * P.ANNUAL
        ys.append(dict(delay=r["delay"], d=d[:len(NET)], flow=r["flow"][:len(NET)], cap=(r["cap"] * s["cap0"])[:len(NET)], vc=((s["carv"] + P.PCU * s["heavy"]) / (s["cap"] * r["cap"] * s["cap0"] / s["cap"]))[:len(NET)],
                       tr=(r["time"] / 3600 / (s["len"] / 1000 / s["kmh"]))[:len(NET)],
                       rej=[a for a in r["actions"] if not a["ok"]], works=r["works"], open=r["open"]))
    return c.score(), c.result(), ys


def job(x):
    name, seed, plan, off = x
    sc, res, ys = run(seed, plan, off)
    return name, seed, sc, res, ys


def main():
    osm = np.array([-1 if l["osm"] is None else l["osm"] for l in NET])
    sets = {k: np.isin(osm, [o for e in CAT[k].get("mod", []) for o in e["osm"]]) for k in ACC}
    zxy = np.array([[z["x"], z["y"]] for z in ZJ["zones"]])
    mid = np.array([np.mean(l["geom"], 0) for l in NET])
    cen = {a["name"]: zxy[a["zones"]].mean(0) for a in ZJ["growth"]["areas"]}
    geo = {k: dict(links=int(m.sum()), nearest=min(cen, key=lambda n: np.linalg.norm(mid[m].mean(0) - cen[n])),
                   km=round(float(np.linalg.norm(mid[m].mean(0) - cen[min(cen, key=lambda n: np.linalg.norm(mid[m].mean(0) - cen[n]))]) / 1000), 2),
                   cls={c: int(v) for c, v in zip(*np.unique([NET[i]["cls"] for i in np.flatnonzero(m)], return_counts=True))}) for k, m in sets.items()}
    jobs = [(n, 4, p, o) for n, (p, o) in ARMS.items()] + [("nothing", s, [], {}) for s in range(16) if s != 4]
    jobs += [(f"only:{k}", s, only(4, k), {}) for s in (4, 14) for k in ACC]
    with Pool(8) as pool:
        out = pool.map(job, jobs)
    R = {(n, s): (sc, res, ys) for n, s, sc, res, ys in out}
    ser = lambda ys, m=None: [round(float(y["delay"] if m is None else y["d"][m].sum()) / 1e6, 3) for y in ys]
    nb = R[("nothing", 4)][2]
    arms = {}
    for n in ARMS:
        sc, res, ys = R[(n, 4)]
        arms[n] = dict(score=sc, delay_mh=res["delay"] / 1e6, rejected=sum(len(y["rej"]) for y in ys), delay=ser(ys), delay_lep=ser(ys, sets[L]), delay_ota=ser(ys, sets[O]),
                       flow_lep=[round(float(y["flow"][sets[L]].sum()) / 1e3, 1) for y in ys], flow_ota=[round(float(y["flow"][sets[O]].sum()) / 1e3, 1) for y in ys],
                       cap_lep=[round(float(y["cap"][sets[L]].sum()) / 1e3, 1) for y in ys], cap_ota=[round(float(y["cap"][sets[O]].sum()) / 1e3, 1) for y in ys],
                       works=[y["works"] for y in ys], open=[y["open"] for y in ys], vs_nothing=[round(float(a["delay"] - b["delay"]) / 1e6, 3) for a, b in zip(ys, nb)])
    tot = lambda n, s=4: sum(y["d"] for y in R[(n, s)][2])
    dd = tot("swap") - tot("base")
    top = [dict(name=NET[i]["name"], cls=NET[i]["cls"], osm=int(osm[i]), set=[k for k in ACC if sets[k][i]], swap_minus_base_mh=round(float(dd[i]) / 1e6, 3),
                base_mh=round(float(tot("base")[i]) / 1e6, 3), swap_mh=round(float(tot("swap")[i]) / 1e6, 3)) for i in np.argsort(-dd)[:15]]
    share = {s: {k: round(float(tot("nothing", s)[sets[k]].sum() / tot("nothing", s).sum()), 3) for k in ACC} for s in (4, 14)}
    single = {s: {k: R[(f"only:{k}", s)][0] for k in ACC} for s in (4, 14)}
    B = 467472478
    bott = {}
    for s in range(16):
        ys, d = R[("nothing", s)][2], tot("nothing", s)
        w = int(np.argmax(ys[14]["d"]))
        bott[s] = dict(score_nothing=R[("nothing", s)][0], way_share=round(float(d[osm == B].sum() / d.sum()), 3), ota_share=round(float(d[sets[O]].sum() / d.sum()), 3),
                       top_link=NET[w]["name"], top_osm=int(osm[w]), top_link_share_2040=round(float(ys[14]["d"][w] / ys[14]["d"].sum()), 3),
                       top_vc_2040=round(float(ys[14]["vc"][w]), 2), top_time_ratio_2040=round(float(ys[14]["tr"][w]), 1), top_len_m=NET[w]["len"], top_lanes=NET[w]["lanes"])
    for n in ("base", "swap", "nothing"):
        ys = R[(n, 4)][2]
        k = np.flatnonzero(osm == B)
        arms[n].update(bott_delay=ser(ys, osm == B), bott_vc=[round(float(y["vc"][k].max()), 2) for y in ys], bott_time_ratio=[round(float(y["tr"][k].max()), 1) for y in ys])
    meta = dict(script="paper/figs/seed4.py", command="uv run python paper/figs/seed4.py", seeds=list(range(16)), date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                machine=f"{platform.node()} {platform.machine()} {open('/proc/cpuinfo').read().split('model name')[1].split(chr(10))[0].strip(': \t')}")
    (ROOT / "evidence/seed4.json").write_text(json.dumps(dict(meta=meta, arms=arms, top=top, nothing_share=share, single_2030=single, geo=geo, bottleneck=bott), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
