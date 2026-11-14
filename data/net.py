import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
import osmium
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from data.geo import OUT, PBF, local

ROADS = ["motorway", "trunk", "primary", "secondary", "tertiary"]
LANES = {"motorway": 2, "trunk": 2, "primary": 1, "secondary": 1, "tertiary": 1, "ramp": 1}
KMH = {"motorway": 100, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 40, "ramp": 50}
CAP = {"motorway": 2000, "trunk": 1800, "primary": 1100, "secondary": 900, "tertiary": 700, "ramp": 1400}
PLACES = {"suburb", "quarter", "village", "town"}
SPACING = 2400
DIRS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]


def num(v):
    m = re.match(r"\d+", v or "")
    return int(m.group()) if m else None


def to_local(g):
    return shapely.transform(g, lambda c: np.column_stack(local(c[:, 0], c[:, 1])))


def read():
    munis, places, ways = {}, [], []
    wkb = osmium.geom.WKBFactory()
    fp = osmium.FileProcessor(PBF).with_locations().with_areas(osmium.filter.KeyFilter("boundary"))
    for o in fp:
        t = o.tags
        if o.is_area():
            if t.get("boundary") == "administrative" and t.get("admin_level") == "8" and not o.from_way():
                munis[t.get("name")] = to_local(shapely.from_wkb(wkb.create_multipolygon(o)))
        elif o.is_node():
            if t.get("place") in PLACES and "name" in t:
                places.append((*local(o.location.lon, o.location.lat), t["name"]))
        elif o.is_way():
            hw = t.get("highway", "")
            if hw.removesuffix("_link") in ROADS:
                xy = [local(n.lon, n.lat) for n in o.nodes]
                ways.append((o.id, "ramp" if hw.endswith("_link") else hw, dict(t), [n.ref for n in o.nodes], xy))
    return munis, places, ways


def attrs(cls, t, fwd):
    one = t.get("oneway") in ("yes", "1", "true", "-1") or t.get("junction") in ("roundabout", "circular")
    side = "forward" if fwd else "backward"
    lanes = num(t.get(f"lanes:{side}")) or (num(t.get("lanes")) and (num(t.get("lanes")) if one else num(t.get("lanes")) // 2)) or LANES[cls]
    ms = t.get(f"maxspeed:{side}") or t.get("maxspeed", "")
    kmh = num(ms) or {"FI:urban": 50, "FI:rural": 80}.get(ms) or KMH[cls]
    ref = (t.get("ref") or "").split(";")[0]
    return {"lanes": lanes, "kmh": kmh, "cap": CAP[cls] * lanes, "cls": cls, "name": t.get("name", ""), "ref": ref, "bridge": t.get("bridge", "no") not in ("no", "")}


def build(region, ways):
    use = Counter(i for w in ways for i in set(w[3]))
    shapely.prepare(region)
    node, xy, links, ends = {}, [], [], []

    def nid(i, p):
        return node[i] if i in node else (node.setdefault(i, len(xy)), xy.append(p))[0]

    for wid, cls, t, ids, pts in ways:
        inside = shapely.contains_xy(region, *np.array(pts).T)
        one = t.get("oneway") in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular")
        rev = t.get("oneway") == "-1"
        start = None
        for k, i in enumerate(ids):
            if not inside[k]:
                if start is not None:
                    ends.append((ids[k - 1], cls, t))
                start = None
            elif start is None:
                if k:
                    ends.append((i, cls, t))
                start = k
            elif use[i] > 1 or k == len(ids) - 1 or not inside[k + 1]:
                a, b = nid(ids[start], pts[start]), nid(ids[k], pts[k])
                geom = pts[start:k + 1]
                if not rev:
                    links.append({"a": a, "b": b, "osm": wid, "geom": geom, **attrs(cls, t, True)})
                if rev or not one:
                    links.append({"a": b, "b": a, "osm": wid, "geom": geom[::-1], **attrs(cls, t, not one)})
                start = k
    return node, xy, links, ends


def merge_ends(node, xy, links, ends):
    rep = {}
    cut = [(node[i], t.get("ref") or t.get("name")) for i, cls, t in ends if cls in ROADS[:3] and i in node]
    for k, (n, key) in enumerate(cut):
        for m, key2 in cut[:k]:
            if key2 == key and math.dist(xy[n], xy[rep.get(m, m)]) < 200:
                rep[n] = rep.get(m, m)
                break
    for l in links:
        for e in "ab":
            if l[e] in rep:
                l[e] = rep[l[e]]
                l["geom"] = [xy[l["a"]]] + l["geom"][1:] if e == "a" else l["geom"][:-1] + [xy[l["b"]]]
    return [l for l in links if l["a"] != l["b"]], {rep.get(n, n) for n, _ in cut}


def strong(xy, links):
    n = len(xy)
    g = coo_matrix((np.ones(len(links)), ([l["a"] for l in links], [l["b"] for l in links])), shape=(n, n))
    _, lab = connected_components(g, directed=True, connection="strong")
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    new = {int(o): k for k, o in enumerate(keep)}
    out = [l | {"a": new[l["a"]], "b": new[l["b"]]} for l in links if l["a"] in new and l["b"] in new]
    return [xy[i] for i in keep], out, new


def segments(links, places):
    ptree, pname = cKDTree([p[:2] for p in places]), [p[2] for p in places]
    key = [l["name"] or l["ref"] or None for l in links]
    mid = np.array([np.mean(l["geom"], 0) for l in links])
    anchors, akey = [], []
    for k in sorted({k for k in key if k}):
        pts = np.array([p for l, kk in zip(links, key) if kk == k for p in l["geom"]])
        c = pts - pts.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        mine = []
        for p in pts[np.argsort(c @ axis)]:
            if not mine or min(math.dist(p, q) for q in mine) > SPACING:
                mine.append(p)
        anchors += mine
        akey += [k] * len(mine)
    atree = cKDTree(anchors := np.array(anchors))
    seg = []
    for i, k in enumerate(key):
        cand = [j for j in atree.query_ball_point(mid[i], SPACING * 1.5) if akey[j] == k] if k else []
        seg.append(min(cand, key=lambda j: math.dist(mid[i], anchors[j])) if cand else int(atree.query(mid[i])[1]))
    ids = {a: s for s, a in enumerate(sorted(set(seg)))}
    out, names = [], Counter()
    for a, s in ids.items():
        mem = [i for i in range(len(links)) if seg[i] == a]
        pts = np.array([p for i in mem for p in links[i]["geom"]])
        top = Counter((links[i]["cls"], links[i]["name"] or (f"Road {links[i]['ref']}" if links[i]["ref"] else "")) for i in mem for _ in links[i]["geom"])
        cls = min({c for c, _ in top}, key=(ROADS + ["ramp"]).index)
        name = (Counter({n: v for (c, n), v in top.items() if n}).most_common(1) or [("Ramp",)])[0][0]
        c = pts - pts.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        axis *= np.sign(axis[np.argmax(abs(axis))])
        p, q = pts[np.argmin(c @ axis)], pts[np.argmax(c @ axis)]
        a1, a2 = pname[ptree.query(p)[1]], pname[ptree.query(q)[1]]
        label = f"{name} {a1} to {a2}" if a1 != a2 else f"{name} at {a1}"
        names[label] += 1
        out.append({"id": s, "name": label if names[label] == 1 else f"{label} {names[label]}", "cls": cls, "links": mem})
        for i in mem:
            links[i]["seg"] = s
    return out


def gates(xy, links, cut, ways, places, centre):
    local_names = {p[2] for p in places}
    signs = [(w[4][-1], np.subtract(w[4][-1], w[4][0]), w[2]["destination"].split(";")) for w in ways if "destination" in w[2]]
    out = []
    for n in sorted(cut):
        ls = [l for l in links if n in (l["a"], l["b"]) and l["cls"] in ROADS[:3]]
        if not ls:
            continue
        u = (np.array(xy[n]) - centre) / np.linalg.norm(np.array(xy[n]) - centre)
        votes = Counter(s for q, v, names in signs if math.dist(q, xy[n]) < 10000 and v @ u > 0.7 * np.linalg.norm(v)
                        for s in names if s not in local_names and not s.startswith("Kehä"))
        d = DIRS[round(math.degrees(math.atan2(u[1], u[0])) / 45) % 8]
        road = Counter(l["name"] or l["ref"] for l in ls).most_common(1)[0][0]
        out.append({"node": n, "name": f"{road} {d}" + (f" ({votes.most_common(1)[0][0]})" if votes else "")})
    return out


def main():
    munis, places, ways = read()
    home = shapely.union(munis["Espoo"], munis["Kauniainen"])
    node, xy, links, ends = build(home.buffer(2500), ways)
    links, cut = merge_ends(node, xy, links, ends)
    xy, links, new = strong(xy, links)
    for l in links:
        l["geom"] = [[round(x, 1), round(y, 1)] for x, y in l["geom"]]
        l["len"] = round(shapely.LineString(l["geom"]).length, 1)
    segs = segments(links, places)
    gs = gates(xy, links, {new[n] for n in cut if n in new}, ways, places, np.array(home.centroid.coords[0]))
    keys = ["a", "b", "len", "lanes", "kmh", "cap", "cls", "name", "ref", "osm", "seg", "bridge", "geom"]
    net = {"nodes": [[round(x, 1), round(y, 1)] for x, y in xy], "links": [{k: l[k] for k in keys} for l in links], "segs": segs, "gates": gs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "net.json").write_text(json.dumps(net, ensure_ascii=False, separators=(",", ":")))
    print(len(xy), "nodes", len(links), "links", len(segs), "segs", len(gs), "gates")


if __name__ == "__main__":
    main()
