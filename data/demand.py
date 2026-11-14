import csv, io, json, zipfile

import numpy as np
import osmium
import requests
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import STRtree, points, unary_union
from shapely.affinity import translate
from shapely.geometry import LineString, Point, shape

from data.geo import ORIGIN, OUT, PBF, RAW, local

WFS = "service=WFS&version=2.0.0&request=GetFeature&outputFormat=application/json&srsName=EPSG:3067&typeNames="
HSY = "https://kartta.hsy.fi/geoserver/wfs?" + WFS
PAAVO = "https://geo.stat.fi/geoserver/postialue/wfs?" + WFS + "postialue:pno_tilasto_2026&CQL_FILTER=kunta%20IN%20('049','235')"
TMS = "https://tie.digitraffic.fi/api/tms/v1/"
GTFS = "https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip"
LAM_DAYS, GTFS_DAY, GTFS_WD = (258, 259, 260), "20261117", "tuesday"
DAILY, PT_SHARE = 3.2 * 0.60 * 0.92, 20 / 60
OCC, WALK, TERM, BT = 1.15, 4.5 / 3.6 / 1.3, 4.0, 0.05
POP_ADD, JOBS_RATE, YEARS = 6500, (220000 / 135386) ** (1 / 37) - 1, 20
AREAS = [
    ("Finnoo", 24.7121, 60.1586, (8000, 13000), (500, 1500)),
    ("Kera", 24.7538, 60.2146, (9000, 16000), (0, 1500)),
    ("Vermo", 24.8421, 60.2143, (3000, 5500), (0, 500)),
    ("Kiviruukki", 24.6347, 60.1638, (5000, 9000), (1500, 3000)),
    ("Espoonlahti", 24.6559, 60.1498, (2000, 5000), (200, 800)),
    ("Leppävaara", 24.8141, 60.2197, (4000, 8000), (2000, 5000)),
    ("Otaniemi and Keilaniemi", 24.8320, 60.1810, (3000, 7000), (5000, 12000)),
    ("Espoon keskus", 24.6560, 60.2050, (3000, 8000), (500, 2000)),
    ("Kauklahti", 24.5962, 60.1915, (2000, 6000), (0, 800)),
    ("Hista", 24.5550, 60.2500, (0, 8000), (0, 3000)),
]


def fetch(name, url):
    p = RAW / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, headers={"Digitraffic-User": "espoo-sim", "Accept-Encoding": "gzip"}, timeout=900)
        p.write_bytes(r.content if r.ok else b"")
    return p.read_bytes()


def geo(name, url):
    return [(translate(shape(f["geometry"]), -ORIGIN[0], -ORIGIN[1]), f["properties"]) for f in json.loads(fetch(name, url))["features"]]


def locate(polys, xy):
    out = np.full(len(xy), -1)
    i, j = STRtree(polys).query(points(xy), predicate="within")
    out[i] = j
    return out


def pdist(p, q):
    return np.hypot(*(p[:, None] - q[None]).transpose(2, 0, 1))


def split(idx, xy, a, cap):
    if a[idx].sum() <= cap or len(idx) < 2:
        return [idx]
    w = a[idx] + 1e-6
    c = xy[idx] - np.average(xy[idx], axis=0, weights=w)
    o = np.argsort(c @ np.linalg.eigh(np.cov(c.T, aweights=w))[1][:, -1])
    k = int(np.clip(np.searchsorted(np.cumsum(w[o]), w.sum() / 2), 1, len(idx) - 1))
    return split(idx[o[:k]], xy, a, cap) + split(idx[o[k:]], xy, a, cap)


def zones(gates, nodes):
    allp = geo("hsy_pien.json", HSY + "taustakartat_ja_aluejaot:seutukartta_pien_2021")
    muni = {int(k): np.array(unary_union([g for g, p in allp if p["kunta"] == k]).centroid.coords[0]) for k in {p["kunta"] for _, p in allp}}
    pien = [(g, p) for g, p in allp if p["kunta"] in ("049", "235")]
    pc, jc = (geo(f"hsy_{n}.json", HSY + "asuminen_ja_maankaytto:" + layer) for n, layer in (("pop", "Vaestotietoruudukko_2025"), ("jobs", "Tyopaikkaruudukko_2024")))
    post = geo("paavo.json", PAAVO)
    xy = np.array([g.centroid.coords[0] for g, _ in pc + jc])
    w = np.array([p["tphklkm"] if p["tphklkm"] > 0 else 0.5 * p["tplkm"] for _, p in jc], float)
    pj = locate([g for g, _ in post], xy[len(pc):])
    tot = np.array([max(p["tp_tyopy"], 0) for _, p in post], float)
    fr = np.array([sum(max(p[k], 0) for k in ("tp_c_teol", "tp_f_rake", "tp_g_kaup", "tp_h_kulj")) for _, p in post]) / np.maximum(tot, 1)
    ws = np.bincount(pj[pj >= 0], w[pj >= 0], len(post))
    col = np.zeros((len(xy), 3))
    col[: len(pc), 0] = [p["asukkaita"] for _, p in pc]
    col[len(pc):, 1] = np.where(pj >= 0, tot[pj] * w / np.maximum(ws[pj], 1e-9), 0)
    col[len(pc):, 2] = np.where(pj >= 0, fr[pj], 0) * col[len(pc):, 1]
    zi = locate([g for g, _ in pien], xy)
    act = col[:, 0] + col[:, 1]
    places = [(o.tags["name"], *local(o.location.lon, o.location.lat)) for o in osmium.FileProcessor(PBF, osmium.osm.NODE).with_filter(osmium.filter.KeyFilter("place"))
              if o.tags["place"] in ("suburb", "quarter", "neighbourhood") and "name" in o.tags]
    pl = np.array([p[1:] for p in places])
    out, fz = [], []
    for k, (g, p) in enumerate(pien):
        for idx in split(np.flatnonzero((zi == k) & (act > 0)), xy, act, act[zi >= 0].sum() / 140):
            if len(idx):
                c = np.average(xy[idx], axis=0, weights=act[idx])
                near = [places[i][0] for i in np.argsort(np.hypot(*(pl - c).T))[:5] if g.contains(Point(pl[i]))]
                out.append({"id": len(out), "name": near[0] if near else p["nimi"] or "Kauniainen", "x": round(c[0], 1), "y": round(c[1], 1),
                            "pop": int(round(col[idx, 0].sum())), "jobs": int(round(col[idx, 1].sum())), "ext": False})
                fz.append(col[idx, 2].sum())
    out += [{"id": len(out) + i, "name": g["name"], "x": nodes[g["node"]][0], "y": nodes[g["node"]][1], "pop": 0, "jobs": 0, "ext": True} for i, g in enumerate(gates)]
    return out, np.array(fz + [0.0] * len(gates)), unary_union([g for g, _ in pien]).buffer(2500), muni


def counts(area, muni):
    out, hours = [], np.zeros(24)
    for f in json.loads(fetch("tms_stations.json", TMS + "stations"))["features"]:
        xy = np.array(local(*f["geometry"]["coordinates"][:2]))
        if not area.contains(Point(xy)):
            continue
        m = json.loads(fetch(f"tms/{f['id']}.json", TMS + f"stations/{f['id']}"))["properties"]
        c, k = np.zeros((2, 2)), 0
        for d in LAM_DAYS:
            raw = fetch(f"tms/{m['tmsNumber']}_{d}.csv", TMS + f"history/raw/lamraw_{m['tmsNumber']}_26_{d}.csv")
            a = np.loadtxt(io.BytesIO(raw), delimiter=";", usecols=(3, 9, 10, 12), dtype=int, ndmin=2) if raw else np.zeros((0, 4), int)
            a = a[a[:, 3] == 0]
            if (a[:, 0] == 7).sum() > 100:
                k += 1
                hours += np.bincount(a[:, 0], minlength=24)
                for s in (1, 2):
                    v = a[(a[:, 0] == 7) & (a[:, 1] == s), 2]
                    c[s - 1] += len(v), np.isin(v, (2, 3, 4, 5)).sum()
        if not k or (b := m["bearing"]) is None:
            continue
        for code, sgn in ((m["direction1MunicipalityCode"], 1), (m["direction2MunicipalityCode"], -1)):
            if code in muni and code != m["municipalityCode"]:
                b += 180 * (sgn * np.dot([np.sin(np.radians(b)), np.cos(np.radians(b))], muni[code] - xy) < 0)
                break
        out += [{"id": f["id"], "name": m["names"].get("en", m["name"]), "x": round(xy[0], 1), "y": round(xy[1], 1), "bearing": int((b + 180 * s) % 360),
                 "veh": int(round(c[s, 0] / k)), "heavy": round(c[s, 1] / c[s, 0], 3), "_r": str(m["roadAddress"]["roadNumber"])} for s in (0, 1) if c[s, 0]]
    return out, hours[7] / hours.sum()


def transit(zxy, lo, hi):
    z = zipfile.ZipFile(io.BytesIO(fetch("hsl.zip", GTFS)))
    rd = lambda f: csv.DictReader(io.TextIOWrapper(z.open(f), "utf-8-sig"))
    svc = {r["service_id"] for r in rd("calendar.txt") if r[GTFS_WD] == "1" and r["start_date"] <= GTFS_DAY <= r["end_date"]}
    for r in rd("calendar_dates.txt"):
        if r["date"] == GTFS_DAY:
            (svc.add if r["exception_type"] == "1" else svc.discard)(r["service_id"])
    trips = {r["trip_id"] for r in rd("trips.txt") if r["service_id"] in svc}
    stops, hub = {}, {}
    for r in rd("stops.txt"):
        x, y = local(float(r["stop_lon"]), float(r["stop_lat"]))
        if r["location_type"] == "0" and lo[0] < x < hi[0] and lo[1] < y < hi[1]:
            stops[r["stop_id"]] = hub.setdefault(r["parent_station"].strip() or r["stop_id"], (len(hub), x, y))
    ride, deps, cont, prev = {}, {}, set(), (None,)
    sec = lambda t: int(t[:-6]) * 3600 + int(t[-5:-3]) * 60 + int(t[-2:])
    for r in csv.reader(io.TextIOWrapper(z.open("stop_times.txt"), "utf-8-sig")):
        if r[0] not in trips or r[3] not in stops:
            prev = (None,)
            continue
        dep, seg = sec(r[2]), None
        if prev[0] == r[0] and 23400 <= prev[2] < 34200:
            seg = stops[prev[1]][0], stops[r[3]][0]
            ride.setdefault(seg, []).append(sec(r[1]) - prev[2])
            deps[seg] = deps.get(seg, 0) + (25200 <= prev[2] < 32400)
            if prev[3]:
                cont.add((prev[3], seg))
        prev = (r[0], r[3], dep, seg)
    sxy = np.array([v[1:] for v in hub.values()])
    S, ix = len(hub), {sg: len(hub) + i for i, sg in enumerate(ride)}
    rt = {sg: max(np.median(v), 1.0) for sg, v in ride.items()}
    e = [(ix[a], ix[b], rt[b]) for a, b in cont]
    for sg, n in ix.items():
        e.append((n, sg[1], 0.0))
        if deps[sg]:
            e.append((sg[0], n, min(3600.0 / deps[sg], 1800.0) + rt[sg]))
    t = STRtree(points(sxy))
    i, j = t.query(points(sxy), predicate="dwithin", distance=300)
    e += [(a, b, np.hypot(*(sxy[a] - sxy[b])) / WALK) for a, b in zip(i, j) if a != b]
    Z, base = len(zxy), S + len(ix)
    for k, c in enumerate(zxy):
        near = t.query(Point(c), predicate="dwithin", distance=1000)
        for s in near if len(near) >= 3 else np.argsort(np.hypot(*(sxy - c).T))[:3]:
            w = 60 + np.hypot(*(sxy[s] - c)) / WALK
            e += [(base + k, s, w), (s, base + Z + k, w)]
    a, b, w = map(np.array, zip(*e))
    pt = dijkstra(coo_matrix((w, (a, b)), shape=(base + 2 * Z,) * 2).tocsr(), indices=base + np.arange(Z))[:, base + Z:]
    return np.minimum(pt, pdist(zxy, zxy) / WALK) / 60


@njit(cache=True)
def tree(tt, pred, ip, ix, ed, lens, zn, dem):
    flow, dist = np.zeros(len(lens)), np.zeros((len(zn), len(zn)))
    for o in range(len(zn)):
        order, load, dd, eo = np.argsort(tt[o]), np.zeros(tt.shape[1]), np.zeros(tt.shape[1]), np.full(tt.shape[1], -1)
        for d in range(len(zn)):
            load[zn[d]] += dem[o, d]
        for n in order:
            p = pred[o, n]
            if p >= 0:
                for k in range(ip[p], ip[p + 1]):
                    if ix[k] == n:
                        eo[n] = ed[k]
                dd[n] = dd[p] + lens[eo[n]]
        for n in order[::-1]:
            if eo[n] >= 0:
                flow[eo[n]] += load[n]
                load[pred[o, n]] += load[n]
        dist[o] = dd[zn]
    return flow, dist


def furness(P, A, F):
    T = F * np.outer(P, A)
    for _ in range(200):
        T *= (P / np.maximum(T.sum(1), 1e-12))[:, None]
        T *= (A / np.maximum(T.sum(0), 1e-12))[None, :]
    return T


def main():
    net = json.loads((OUT / "net.json").read_text())
    nodes, L, gates = np.array(net["nodes"]), net["links"], net["gates"]
    zs, fz, region, muni = zones(gates, net["nodes"])
    ext, zxy = np.array([z["ext"] for z in zs]), np.array([[z["x"], z["y"]] for z in zs])
    pop, jobs = np.array([z["pop"] for z in zs], float), np.array([z["jobs"] for z in zs], float)
    cs, phf = counts(region.buffer(6000), muni)
    a, b = np.array([l["a"] for l in L]), np.array([l["b"] for l in L])
    lens, kmh = np.array([[l["len"], l["kmh"]] for l in L], float).T
    tl = lens / kmh * 3.6
    o = np.lexsort((tl, b, a))
    keep = o[np.r_[True, (np.diff(a[o]) != 0) | (np.diff(b[o]) != 0)]]
    g = coo_matrix((tl[keep], (a[keep], b[keep])), shape=(len(nodes),) * 2).tocsr()
    eid = coo_matrix((keep + 1.0, (a[keep], b[keep])), shape=g.shape).tocsr()
    eix = eid.data.astype(np.int64) - 1
    ln = np.unique(np.r_[a, b][np.tile([l["cls"] not in ("motorway", "trunk", "ramp") for l in L], 2)])
    zn = ln[np.argmin(pdist(zxy, nodes[ln]), axis=1)]
    zn[ext] = [gt["node"] for gt in gates]
    tt, pred = dijkstra(g, indices=zn, return_predecessors=True)
    tc = np.nan_to_num(tt[:, zn] / 60, posinf=999) + TERM
    run = lambda dem: tree(tt, pred, eid.indptr, eid.indices, eix, lens, zn, dem)
    dist = run(np.zeros((len(zs), len(zs))))[1]
    pt = np.minimum(transit(zxy, zxy.min(0) - 3000, zxy.max(0) + 3000), 300)
    head = np.degrees(np.arctan2(*(nodes[b] - nodes[a]).T)) % 360
    st = STRtree(lines := [LineString(l["geom"]) for l in L])
    for c in cs:
        p = Point(c["x"], c["y"])
        near = [i for i in st.query(p, predicate="dwithin", distance=300) if L[i]["ref"] == c["_r"] and abs((head[i] - c["bearing"] + 180) % 360 - 180) < 60]
        c["_l"] = min(near, key=lambda i: lines[i].distance(p)) if near else -1
    gio, ip, ie = np.zeros((4, len(gates))), ~ext, ext
    for k, gt in enumerate(gates):
        gx = nodes[gt["node"]]
        refs = {l["ref"] for l in L if gt["node"] in (l["a"], l["b"]) and l["ref"]}
        cand = sorted([c for c in cs if (c["_r"] in refs or not refs) and np.hypot(c["x"] - gx[0], c["y"] - gx[1]) < 6000], key=lambda c: np.hypot(c["x"] - gx[0], c["y"] - gx[1]))
        for c in [c for c in cand if c["id"] == cand[0]["id"]]:
            inb = np.dot([np.sin(np.radians(c["bearing"])), np.cos(np.radians(c["bearing"]))], zxy[ip].mean(0) - gx) > 0
            gio[[1 - inb, 3 - inb], k] += c["veh"] * (1 - c["heavy"]), c["veh"] * c["heavy"]
        if not cand:
            gio[:, k] = np.array([0.5, 0.5, 0.02, 0.02]) * max(l["cap"] for l in L if gt["node"] in (l["a"], l["b"]))
        print(f"gate {gt['name']}: cars in/out, heavy in/out {gio[:, k].round().tolist()} from {cand[0]['name'] if cand else 'capacity'}")
    li, oc, oh = (np.array(v) for v in zip(*[(c["_l"], c["veh"] * (1 - c["heavy"]), c["veh"] * c["heavy"]) for c in cs if c["_l"] >= 0]))
    far = ~ext[:, None] | ~ext[None] | (pdist(zxy, zxy) > 10000)
    np.fill_diagonal(far, False)
    rate, inner, act = DAILY * phf, ~ext[:, None] & ~ext[None], pop + jobs

    def model(beta, asc=0.0):
        F = np.exp(-beta * tc) * far
        for _ in range(6):
            pc = 1 / (1 + np.exp(-(asc + BT * (pt - tc))))
            P, A = np.where(ext, 0, rate * pop), np.where(ext, 0, jobs)
            P[ext] = gio[0] * OCC / np.maximum((F[ext] * pc[ext] * act).sum(1) / np.maximum((F[ext] * act).sum(1), 1e-9), 0.05)
            A[ext] = gio[1] * OCC / np.maximum((F[:, ext] * pc[:, ext] * act[:, None]).sum(0) / np.maximum((F[:, ext] * act[:, None]).sum(0), 1e-9), 0.05)
            A[~ext] *= (P.sum() - A[ext].sum()) / A[~ext].sum()
            T = furness(P, A, F)
            for _ in range(30):
                asc += 4 * ((T * (1 - pc))[inner].sum() / T[inner].sum() - PT_SHARE)
                pc = 1 / (1 + np.exp(-(asc + BT * (pt - tc))))
        f = run(T * pc / OCC)[0][li]
        return np.sqrt(((f - oc) ** 2).mean()), beta, T, pc, asc, f

    err, beta, T, pc, asc, f = min((model(bt) for bt in (0.03, 0.05, 0.07, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4)), key=lambda r: r[0])
    geh = np.sqrt(2 * (f - oc) ** 2 / (f + oc))
    print(f"peak hour factor {phf:.3f}, rate {rate:.3f} trips/resident/h, beta {beta}/min, asc {asc:.2f}, cars at {len(li)} count links: RMSE {err:.0f} veh/h, GEH<5 {np.mean(geh < 5):.0%}, GEH<10 {np.mean(geh < 10):.0%}, model/obs {f.sum() / oc.sum():.2f}")
    Ft = np.exp(-beta / 2 * tc) * far

    def trucks(k):
        P, A = k * fz, k * fz
        P[ext], A[ext] = gio[2], gio[3]
        H = furness(P, A * P.sum() / A.sum(), Ft)
        return np.sqrt(((run(H)[0][li] - oh) ** 2).mean()), k, H

    herr, hk, H = min((trucks(k) for k in np.linspace(0.002, 0.06, 30)), key=lambda r: r[0])
    print(f"trucks {hk:.4f} veh/h per freight job, total {H.sum():.0f} veh/h, RMSE {herr:.0f}, model/obs {run(H)[0][li].sum() / oh.sum():.2f}")
    print(f"zones {ip.sum()} + {ie.sum()} ext, pop {pop.sum():.0f}, jobs {jobs.sum():.0f}, AM peak person trips {T.sum():.0f}: internal {T[ip][:, ip].sum():.0f}, out {T[ip][:, ie].sum():.0f}, in {T[ie][:, ip].sum():.0f}, through {T[ie][:, ie].sum():.0f}")
    print(f"mean car-path length {(T * dist).sum() / T.sum() / 1000:.1f} km, internal {(T * dist)[inner].sum() / T[inner].sum() / 1000:.1f} km; car {(T * pc).sum() / T.sum():.0%}, internal car {(T * pc)[inner].sum() / T[inner].sum():.0%}")
    p0, j0, areas, used = pop[ip].sum(), jobs[ip].sum(), [], set()
    rt = lambda cur, add: round(((cur + add) / cur) ** (1 / YEARS) - 1, 4)
    for name, lon, lat, pa, ja in AREAS:
        d = np.where(ip, np.hypot(*(zxy - local(lon, lat)).T), np.inf)
        order = [int(i) for i in np.argsort(d) if i not in used and np.isfinite(d[i])]
        zi = order[:max(np.sum(d[order] < 1200), np.searchsorted(np.cumsum(pop[order]), 2000) + 1)]
        used |= set(zi)
        cp, cj = max(pop[zi].sum(), 500), max(jobs[zi].sum(), 500)
        areas.append({"name": name, "zones": zi, "pop": [rt(cp, v) for v in pa], "jobs": [rt(cj, v) for v in ja]})
    rest = np.array([i for i in np.flatnonzero(ip) if i not in used])
    end = lambda col, key: sum(max(col[ar["zones"]].sum(), 500) * (1 + np.mean(ar[key])) ** YEARS for ar in areas)
    bp = (((p0 + YEARS * POP_ADD) - end(pop, "pop")) / pop[rest].sum()) ** (1 / YEARS) - 1
    bj = ((j0 * (1 + JOBS_RATE) ** YEARS - end(jobs, "jobs")) / jobs[rest].sum()) ** (1 / YEARS) - 1
    print(f"growth outside areas: pop {bp:.4f}, jobs {bj:.4f}; city pop {((p0 + YEARS * POP_ADD) / p0) ** (1 / YEARS) - 1:.4f}, jobs {JOBS_RATE:.4f}")
    (OUT / "zones.json").write_text(json.dumps({"zones": zs, "growth": {"pop": round(bp, 4), "jobs": round(bj, 4), "areas": areas}}, ensure_ascii=False))
    (OUT / "counts.json").write_text(json.dumps([{k: v for k, v in c.items() if k[0] != "_"} for c in cs if region.contains(Point(c["x"], c["y"]))], ensure_ascii=False))
    for name, m in (("od", T), ("truck", H), ("pt", pt)):
        np.save(OUT / f"{name}.npy", m.astype(np.float32))


if __name__ == "__main__":
    main()
