import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from math import floor

import numpy as np
import osmium
import requests
import shapefile
import shapely
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.geometry.polygon import orient
from shapely.ops import substring

from data.geo import BBOX, ORIGIN, PBF, RAW, VIZ, local

S = 1000
TOL = 0.3
HYPE = local(24.6306, 60.1568)
LAND = RAW / "land" / "land-polygons-split-4326" / "land_polygons.shp"
WFS = "https://kartat.espoo.fi/teklaogcweb/wfs.ashx?service=WFS&version=1.1.0&request=GetFeature&typeName=bldg:building_lod1"
LOD1 = RAW / "espoo_lod1.npy"
LAND_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"

BKINDS = ["residential", "apartments", "commercial", "industrial", "public", "garage", "other"]
BMAP = {
    "residential": "house detached semidetached_house terrace residential bungalow farm cabin",
    "apartments": "apartments dormitory",
    "commercial": "commercial retail office hotel supermarket kiosk",
    "industrial": "industrial warehouse service hangar factory manufacture storage_tank greenhouse barn farm_auxiliary",
    "public": "public school university college hospital civic government kindergarten church chapel sports_hall sports_centre train_station transportation stadium fire_station library museum",
    "garage": "garage garages carport shed hut roof parking boathouse",
}
BMAP = {v: k for k, vs in BMAP.items() for v in vs.split()}
BH = {"residential": 6, "apartments": 15, "commercial": 10, "industrial": 8, "public": 11, "garage": 3, "other": 5}
GKINDS = ["land", "water", "forest", "park", "grass", "field", "wetland", "sand", "rock", "paved", "rail", "pitch",
          "parking", "playground", "garden", "scrub", "hedge", "sidewalk", "conifer_forest"]
BASE = {"parking": "paved", "playground": "paved", "sidewalk": "paved", "garden": "park", "scrub": "grass", "conifer_forest": "forest"}
GMAP = {
    "natural": {"water": "water", "wood": "forest", "scrub": "scrub", "grassland": "grass", "heath": "grass", "sand": "sand", "beach": "sand", "wetland": "wetland", "bare_rock": "rock", "scree": "rock"},
    "landuse": {"forest": "forest", "grass": "grass", "meadow": "field", "farmland": "field", "orchard": "field", "recreation_ground": "park", "cemetery": "park", "village_green": "park",
                "allotments": "park", "railway": "rail", "reservoir": "water", "basin": "water", "quarry": "sand", "construction": "sand"},
    "leisure": {"park": "park", "garden": "garden", "golf_course": "grass", "pitch": "pitch", "track": "pitch", "swimming_pool": "water", "playground": "playground"},
    "amenity": {"parking": "parking"}, "place": {"square": "paved"}, "highway": {"pedestrian": "sidewalk", "footway": "sidewalk"},
    "area:highway": {"footway": "sidewalk", "pedestrian": "sidewalk", "sidewalk": "sidewalk"},
}
PKINDS = ["crossing", "signal", "lamp", "bus_stop", "bench", "bollard", "pole", "conifer", "broadleaf"]
PMAP = {"natural": {"tree": "tree"}, "highway": {"crossing": "crossing", "traffic_signals": "signal", "street_lamp": "lamp", "bus_stop": "bus_stop"},
        "amenity": {"bench": "bench"}, "barrier": {"bollard": "bollard"}, "power": {"pole": "pole", "tower": "pole"}}
LEAF = {"needleleaved": "conifer", "broadleaved": "broadleaf"}
RKINDS = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "track", "footway", "cycleway", "rail", "subway", "light_rail", "tram", "tunnel"]
RMAP = {"residential": "residential", "unclassified": "residential", "living_street": "residential", "road": "residential", "pedestrian": "footway", "path": "footway",
        "steps": "footway", "bridleway": "footway", "footway": "footway", "cycleway": "cycleway", "service": "service", "track": "track", "busway": "service"}
RLANES = {"motorway": 4, "trunk": 4, "primary": 2, "secondary": 2, "tertiary": 2, "residential": 2}
RAIL = {"rail", "subway", "light_rail", "tram"}


def num(v):
    return float(m[0]) if (m := re.match(r"\d+(\.\d+)?", (v or "").replace(",", "."))) else None


def proj(g):
    return shapely.transform(g, lambda c: np.column_stack(local(c[:, 0], c[:, 1])))


def flat(coords, nd=1):
    return [round(float(v), nd or None) for p in coords for v in p]


def parts(g):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and not p.is_empty]


def rkind(t):
    r, h = t.get("railway"), t.get("highway", "").removesuffix("_link")
    if r in RAIL:
        return "tunnel" if t.get("tunnel") in ("yes", "building_passage") else r
    return h if h in RLANES else RMAP.get(h)


def bkind(t):
    return BMAP.get(t.get("building")) or ("public" if t.get("amenity") in ("school", "kindergarten", "hospital", "place_of_worship") else "other")


def lookup(maps, t):
    return next((m[t.get(key)] for key, m in maps.items() if t.get(key) in m), None)


def gkind(t):
    k = lookup(GMAP, t)
    if k == "sidewalk" and t.get("highway") and t.get("area") != "yes":
        return None
    return "conifer_forest" if k == "forest" and t.get("leaf_type") == "needleleaved" else k


def flags(k, t):
    side = t.get("sidewalk") in ("both", "left", "right", "yes") or (k == "residential" and t.get("sidewalk") is None and t.get("lit") == "yes")
    oneway = t.get("oneway") in ("yes", "1", "-1") or t.get("junction") == "roundabout" or k == "motorway"
    return oneway | 2 * (t.get("bridge", "no") != "no") | 4 * side | 8 * (t.get("lit") == "yes")


class Reader(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.wkb = osmium.geom.WKBFactory()
        self.b, self.g, self.r, self.t, self.p, self.muni = [], [], [], [], [], {}

    def node(self, n):
        if k := lookup(PMAP, n.tags):
            xy = local(n.location.lon, n.location.lat)
            k = LEAF.get(n.tags.get("leaf_type")) if k == "tree" else k
            self.p.append((k, *xy)) if k else self.t.append(xy)

    def way(self, w):
        t, k = w.tags, rkind(w.tags)
        row, hedge = t.get("natural") == "tree_row", t.get("barrier") == "hedge"
        if not (line := (k or row or hedge) and self.geom(self.wkb.create_linestring, w)):
            return
        if hedge:
            self.g += [("hedge", substring(line, a, a + 0.5, normalized=True).buffer(0.75, cap_style="flat")) for a in (0, 0.5) if line.length > 1]
            return
        if row:
            n = max(int(line.length / 8), 1)
            self.t += [(p.x, p.y) for p in line.interpolate(np.linspace(0, line.length, n + 1))]
            return
        lanes = int(num(t.get("lanes")) or (1 if k in RAIL or "_link" in t.get("highway", "") else RLANES.get(k, 1)))
        line = line.reverse() if t.get("oneway") == "-1" else line
        self.r.append((k, lanes, line.simplify(TOL), flags(k, t)))

    def area(self, a):
        t = a.tags
        if t.get("boundary") == "administrative" and t.get("admin_level") == "8" and t.get("name") in ("Espoo", "Kauniainen"):
            self.muni[t["name"]] = self.polys(a)
        if t.get("building", "no") != "no" and t.get("location") != "underground":
            lv = num(t.get("building:levels"))
            hs = [num(t.get("height")), lv and lv * 3.2]
            self.b += [(hs, bkind(t), p.simplify(TOL)) for p in self.polys(a)]
            return
        k = gkind(t)
        if k and t.get("parking") not in ("underground", "multi-storey"):
            self.g += [(k, p) for p in self.polys(a)]

    def polys(self, a):
        return parts(self.geom(self.wkb.create_multipolygon, a) or shapely.Polygon())

    def geom(self, make, o):
        try:
            return proj(shapely.from_wkb(make(o)))
        except Exception:
            return None


def tile(x, y):
    return floor(x / S), floor(y / S)


def region(r):
    xs, ys = local(np.array([BBOX[0], BBOX[2], BBOX[0], BBOX[2]]), np.array([BBOX[1], BBOX[1], BBOX[3], BBOX[3]]))
    (i0, j0), (i1, j1) = tile(xs.min(), ys.min()), tile(xs.max(), ys.max())
    hype = shapely.Point(HYPE)
    return {(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1) if not r or cell(i, j).distance(hype) <= r}


def cell(i, j):
    return box(i * S, j * S, (i + 1) * S, (j + 1) * S)


def land():
    if not LAND.exists():
        z = RAW / "land" / "land.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        z.write_bytes(requests.get(LAND_URL, timeout=600).content)
        zipfile.ZipFile(z).extractall(z.parent)
    shapes = shapefile.Reader(str(LAND)).iterShapes(bbox=BBOX)
    return [("land", p) for s in shapes for p in parts(proj(shape(s.__geo_interface__)))]


def lod1_cell(x, y):
    out = []
    for _, e in ET.iterparse(io.BytesIO(requests.get(f"{WFS}&bbox={x},{y},{x + 2000},{y + 2000}", timeout=600).content)):
        if e.tag.endswith("}Building"):
            h = e.find("{http://www.opengis.net/citygml/building/2.0}measuredHeight")
            p = [a[0].text.split() for a in e if a.get("name") == "sijainti"]
            if h is not None and p:
                out.append((float(p[0][1]), float(p[0][0]), float(h.text)))
            e.clear()
    return out


def lod1():
    if not LOD1.exists():
        e, n = Transformer.from_crs(4326, 3879, always_xy=True).transform([BBOX[0], BBOX[2]], [BBOX[1], BBOX[3]])
        cells = [(x, y) for x in range(int(e[0]) // 2000 * 2000, int(e[1]) + 1, 2000) for y in range(int(n[0]) // 2000 * 2000, int(n[1]) + 1, 2000)]
        with ThreadPoolExecutor(6) as ex:
            rows = np.array([r for c in ex.map(lod1_cell, *zip(*cells)) for r in c])
        x, y = Transformer.from_crs(3879, 3067, always_xy=True).transform(rows[:, 0], rows[:, 1])
        np.save(LOD1, np.column_stack([x - ORIGIN[0], y - ORIGIN[1], rows[:, 2]]))
    return np.load(LOD1)


def heights(b):
    m = lod1()
    pi, bi = shapely.STRtree([p for _, _, p in b]).query(shapely.points(m[:, :2]), predicate="intersects")
    espoo = np.zeros(len(b))
    np.maximum.at(espoo, bi, m[pi, 2])
    src, out = {"height": 0, "espoo": 0, "levels": 0, "kind": 0}, []
    for (hs, k, p), e in zip(b, espoo):
        s, h = next((s, h) for s, h in zip(src, [hs[0], e, hs[1], BH[k]]) if h)
        src[s] += 1
        out.append((round(max(h, 2.5), 1), "apartments" if k == "other" and h >= 12 else k, p))
    return out, src


def split(g, keep):
    x0, y0, x1, y1 = g.bounds
    if x1 - x0 <= S and y1 - y0 <= S:
        return [(tile(g.centroid.x, g.centroid.y), g)]
    keys = [(i, j) for i in range(floor(x0 / S), floor(x1 / S) + 1) for j in range(floor(y0 / S), floor(y1 / S) + 1)]
    return [(k, p) for k in keys if k in keep for p in parts(g.intersection(cell(*k)))]


def angles(p, roads):
    lines = np.array([l for k, _, l, _ in roads if RKINDS.index(k) <= RKINDS.index("service")])
    pts = shapely.points([(x, y) for _, x, y in p])
    pi, li = shapely.STRtree(lines).query_nearest(pts, max_distance=30, all_matches=False)
    d = shapely.line_locate_point(lines[li], pts[pi])
    q0, q1 = (shapely.line_interpolate_point(lines[li], np.maximum(d + o, 0)) for o in (-1, 1))
    a = np.zeros(len(p))
    a[pi] = np.arctan2(shapely.get_y(q1) - shapely.get_y(q0), shapely.get_x(q1) - shapely.get_x(q0))
    return a


def overview(polys, roads, muni):
    polys = [(BASE.get(k, k), p) for k, p in polys if k != "hedge"]
    g = []
    for k in GKINDS:
        ps = shapely.make_valid(np.array([p.simplify(5) for kk, p in polys if kk == k and p.area > 500] or [shapely.Polygon()]))
        u = shapely.union_all(ps)
        for q in parts(u.simplify(15)):
            g += [(k, shapely.Polygon(q.exterior))] + [("land", shapely.Polygon(i)) for i in q.interiors]
    g = sorted((e for e in g if e[1].area > 3000), key=lambda e: -e[1].area)
    major = {}
    [major.setdefault((k, n), []).append(l) for k, n, l, _ in roads if k in ("motorway", "trunk", "primary", "secondary", "rail")]
    r = [[RKINDS.index(k), n, flat(l.simplify(15).coords, 0)] for (k, n), ls in major.items()
         for l in shapely.get_parts(shapely.line_merge(shapely.union_all(ls)))]
    g = [[GKINDS.index(k), flat(orient(q).exterior.coords[:-1], 0)] for k, q in g]
    rim = {k: [flat(p.exterior.simplify(10).coords[:-1], 0) for p in muni.get(n, [])] for k, n in [("border", "Espoo"), ("inner", "Kauniainen")]}
    (VIZ / "overview.json").write_text(json.dumps({"g": g, "r": r, **rim}, separators=(",", ":")))
    print("overview", len(g), "polygons", len(r), "lines")


def main():
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    keep = region(r)
    h = Reader()
    h.apply_file(str(PBF), locations=True)
    tiles = {k: {"b": [], "g": [], "r": [], "t": [], "p": []} for k in keep}
    put = lambda k, layer, item: k in tiles and tiles[k][layer].append(item)
    b, src = heights(h.b)
    for ht, k, p in b:
        put(tile(p.centroid.x, p.centroid.y), "b", [ht, BKINDS.index(k), flat(orient(p).exterior.coords[:-1])])
    polys = land() + h.g
    overview(polys, h.r, h.muni)
    ground = []
    for k, p in polys:
        for key, q in split(p, keep):
            ground += [(key, k, shapely.Polygon(q.exterior).simplify(TOL), 1)]
            ground += [(key, "land", shapely.Polygon(i).simplify(TOL), 0) for i in q.interiors]
    ground.sort(key=lambda e: (-round(e[2].area), e[3]))
    for key, k, q, _ in ground:
        if q.area > 4:
            put(key, "g", [GKINDS.index(k), flat(orient(q).exterior.coords[:-1])])
    for k, lanes, line, f in h.r:
        c = line.interpolate(0.5, normalized=True)
        put(tile(c.x, c.y), "r", [RKINDS.index(k), lanes, flat(line.coords)] + [f] * bool(f))
    for (k, x, y), a in zip(h.p, angles(h.p, h.r)):
        put(tile(x, y), "p", [PKINDS.index(k), round(x, 1), round(y, 1), round(float(a), 2)])
    for x, y in h.t:
        put(tile(x, y), "t", [round(x, 1), round(y, 1)])
    d = VIZ / "tiles"
    d.mkdir(parents=True, exist_ok=True)
    [f.unlink() for f in d.glob("*.json")]
    out = []
    for (i, j), v in sorted(tiles.items()):
        if v["b"] or v["g"] or v["r"]:
            v["t"] = [c for xy in v["t"] for c in xy]
            (d / f"{i}_{j}.json").write_text(json.dumps(v, separators=(",", ":")))
            out.append([i, j])
    meta = {"size": S, "bkinds": BKINDS, "gkinds": GKINDS, "rkinds": RKINDS, "pkinds": PKINDS, "tiles": out}
    (VIZ / "tiles.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(len(out), "tiles", sum(len(v["b"]) for v in tiles.values()), "buildings", src)


if __name__ == "__main__":
    main()
