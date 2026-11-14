import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring

from data.geo import ROOT, VIZ

C = np.array([-1511.7, 1239.2])
S, PX, PAD, N = 96, 0.075, 0.12, 2048
E = S * (1 + 2 * PAD)
RAW, OUT, ASSETS = ROOT / "data/raw/ground", VIZ / "ground", ROOT / "viz/assets/ground"
WMS = "https://kartat.espoo.fi/teklaogcweb/wms.ashx"
MODEL, OPT, CAP = "gpt-image-2.6-flare", {"size": "2048x2048", "quality": "high"}, 88.0
PAL = {"land": (138, 140, 116), "grass": (110, 138, 74), "forest": (86, 80, 56), "sand": (192, 180, 146),
       "water": (44, 64, 82), "rock": (150, 146, 138), "paving": (164, 160, 152), "ballast": (118, 108, 98),
       "path": (112, 112, 114), "asphalt": (78, 80, 84), "walk": (178, 176, 170), "curb": (212, 210, 204),
       "platform": (196, 194, 188), "building": (128, 128, 128), "line": (246, 246, 242), "rail": (52, 46, 42)}
CLS = {k: i for i, k in enumerate(PAL)}
GMAP = {"water": "water", "forest": "forest", "conifer_forest": "forest", "scrub": "forest", "wetland": "forest", "hedge": "forest",
        "park": "grass", "grass": "grass", "garden": "grass", "pitch": "grass", "field": "grass", "sand": "sand",
        "playground": "sand", "rock": "rock", "paved": "paving", "sidewalk": "paving", "parking": "asphalt", "rail": "ballast"}
DRIVE = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
PROMPT = (
    "The first image is an exact top-down orthographic map of a {e:.0f} m square of ground in Kivenlahti, Espoo, Finland, north up, "
    "drawn in flat colour codes. Repaint it as a clean, finely detailed architectural illustration of that ground seen from directly "
    "above, rich in fine material detail like a premium hand-finished isometric city illustration. Keep every boundary, road edge, kerb, lane line, zebra stripe, stop line, lane arrow, parking bay line and building outline exactly "
    "where it is: do not move, add, remove, bend or resize anything. Colour codes: dark grey = asphalt with visible fine aggregate, faint darker tyre tracks, "
    "thin tar-sealed cracks and a few rectangular patches; thin white lines and bars = white road markings, slightly worn (all markings in Finland are white); "
    "light grey bands beside roads = sidewalks of small grey concrete pavers in a regular pattern; very light grey thin bands = granite kerb stones; mid grey with "
    "slightly darker tone = light traffic paths of asphalt; warm light grey areas = paved plazas of paving stones; pale grey "
    "blocks = bus stop platforms with a tactile strip along the kerb; green = mown lawn with subtle colour variation and mowing "
    "stripes; dark olive brown = forest floor seen from above: dark soil, moss patches, fallen leaves, needle litter, boulders and low shrubs, never lawn and no tree crowns; "
    "beige = sand or gravel; dark blue grey = water; brownish grey with dark double lines = railway ballast with rails and "
    "sleepers; olive grey = other ground: paint what the second image shows there (rough grass, gravel, bare earth, flower "
    "beds, small paths); flat mid grey with a dark outline = building footprint: keep it exactly flat uniform grey with no detail. "
    "The second image is a real aerial photograph of the same area for colour and material reference only; geometry must follow "
    "the first image. Soft even overcast daylight, no shadows, no cars, no people, no trees or tree crowns, no text, no labels.")


def load(x0, y0, x1, y1):
    meta = json.loads((VIZ / "tiles.json").read_text())
    g, b, r, p = [], [], [], []
    for i in range(int(x0 // 1000) - 1, int(x1 // 1000) + 1):
        for j in range(int(y0 // 1000) - 1, int(y1 // 1000) + 1):
            f = VIZ / f"tiles/{i}_{j}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            pts = lambda f: np.array(f).reshape(-1, 2)
            g += [(meta["gkinds"][k], Polygon(pts(f))) for k, f in d["g"]]
            b += [Polygon(pts(f)) for _, _, f in d["b"]]
            r += [(meta["rkinds"][e[0]], e[1], LineString(pts(e[2])), e[3] if len(e) > 3 else 0) for e in d["r"]]
            p += [(meta["pkinds"][k], Point(x, y), a) for k, x, y, a in d["p"]]
    return g, b, r, p


def width(k, lanes):
    return max(lanes, 1) * 3.3 + 1 if k in DRIVE else {"service": 3.5, "track": 3, "footway": 2.5, "cycleway": 2.5}.get(k, 3)


def dashes(line, on, off):
    return [substring(line, a, a + on) for a in np.arange(0, line.length, on + off)]


def markings(k, lanes, line, fl):
    if k not in DRIVE:
        return []
    n, half, fast = max(lanes, 1), max(lanes, 1) * 1.65, k in ("motorway", "trunk")
    offs = [-half + 3.3 * i for i in range(1, n)] if fl & 1 else [3.3 * i for i in range(1 - n // 2, n // 2)] if n > 1 else []
    out = [g.buffer(0.08, cap_style="flat") for o in offs for g in dashes(line.offset_curve(o), *((9, 15) if fast else (3, 9)))]
    if DRIVE.index(k) < 4:
        out += [line.offset_curve(s * (half - 0.25)).buffer(0.1, cap_style="flat") for s in (-1, 1)]
    return out


def bars(c, a, along, across, step, span):
    t, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    q = np.array([[-along, -across], [along, -across], [along, across], [-along, across]]) / 2
    return [Polygon(c + o * n + q[:, :1] * t + q[:, 1:] * n) for o in np.arange(-span / 2 + step / 2, span / 2, step)]


def stalls(poly):
    rr = np.array(poly.minimum_rotated_rectangle.exterior.coords[:3])
    e = [rr[1] - rr[0], rr[2] - rr[1]]
    u = max(e, key=np.linalg.norm)
    u, a = u / np.linalg.norm(u), np.arctan2(u[1], u[0])
    v, c, R = np.array([-u[1], u[0]]), np.array(poly.centroid.coords[0]), poly.minimum_rotated_rectangle.length
    lines = []
    for w in np.arange(-R, R, 16):
        lines += [LineString([c + s * u + w * v, c + s * u + (w + 10) * v]) for s in np.arange(-R, R, 2.5)]
        lines.append(LineString([c - R * u + (w + 5) * v, c + R * u + (w + 5) * v]))
    return shapely.unary_union([l.buffer(0.07) for l in lines]).intersection(poly.buffer(-1.2))


def scene(x0, y0):
    g, b, r, p = load(x0 - 20, y0 - 20, x0 + E + 20, y0 + E + 20)
    clip = box(x0 - 20, y0 - 20, x0 + E + 20, y0 + E + 20)
    lay = [("water", clip)] + [(GMAP.get(k, "land"), q) for k, q in g if q.intersects(clip)]
    lay += [("line", stalls(q)) for k, q in g if k == "parking" and q.intersects(clip) and q.area > 150]
    r = [e for e in r if e[2].intersects(clip.buffer(200)) and e[0] != "tunnel"]
    drive = [e for e in r if e[0] in DRIVE]
    tree = shapely.STRtree([e[2] for e in drive])
    sig, cross, cut, zebra = [q for k, q, _ in p if k == "signal"], [q for k, q, _ in p if k == "crossing"], [], []
    for e in drive:
        for end, q in ((0, Point(e[2].coords[0])), (1, Point(e[2].coords[-1]))):
            near = tree.query(q, predicate="dwithin", distance=0.5)
            if len(near) < 3:
                continue
            R = 0.6 * max(width(*drive[i][:2]) for i in near) + 1.5
            cut.append(q.buffer(R))
            R -= 1.2
            k, lanes, ln, fl = e
            ss = [abs(end * ln.length - ln.project(x)) for x in cross if x.distance(ln) < 4]
            R = max([R] + [v + 2.4 for v in ss if v < 30])
            if ln.length < R + 12 or (fl & 1 and not end) or (lanes < 2 and not fl & 1):
                continue
            c = np.array(q.coords[0])
            u = np.array(ln.interpolate(1 if not end else ln.length - 1).coords[0]) - c
            u, half = u / np.linalg.norm(u), max(lanes, 1) * 1.65
            rn = np.array([-u[1], u[0]])
            offs = [-half + 1.65 + 3.3 * i for i in range(lanes)] if fl & 1 else [1.65 + 3.3 * i for i in range(max(lanes // 2, 1))]
            a = np.arctan2(u[1], u[0])
            if any(q.distance(g) < R + 15 for g in sig):
                zebra += bars(c + u * (R + 1.5) + rn * np.mean(offs), a, 0.4, 3.3 * len(offs) - 0.3, 1, 1)
            if len(offs) > 1:
                arrow = np.array([[-0.08, 6], [-0.08, 3.4], [-0.45, 3.4], [0, 2], [0.45, 3.4], [0.08, 3.4], [0.08, 6]])
                zebra += [Polygon(c + (R + 2) * u + o * rn + arrow[:, :1] * rn + arrow[:, 1:] * u) for o in offs]
    for k, q, a in p:
        i = tree.query_nearest(q, max_distance=4)
        if k == "crossing" and len(i):
            zebra += bars(np.array(q.coords[0]), a, 3.5, 0.5, 1.0, width(*drive[i[0]][:2]) - 0.6)
            cut.append(q.buffer(3))
    cut = shapely.unary_union(cut) if cut else Polygon()
    for k, q, a in p:
        if k == "bus_stop" and q.intersects(clip):
            i = tree.query_nearest(q, max_distance=15)
            if len(i):
                ln, w = drive[i[0]][2], width(*drive[i[0]][:2])
                o = ln.interpolate(ln.project(q))
                d = np.array(q.coords[0]) - np.array(o.coords[0])
                d = d / (np.linalg.norm(d) or 1) if np.linalg.norm(d) > 0.3 else np.array([np.sin(a), -np.cos(a)])
                c = np.array(o.coords[0]) + d * (w / 2 + 1.6)
                lay += [("curb", bars(c, a, 18.6, 3.2, 1, 1)[0]), ("platform", bars(c, a, 18, 2.6, 1, 1)[0])]
    rank = lambda e: (bool(e[3] & 2), -DRIVE.index(e[0]) if e[0] in DRIVE else -9)
    for bridge in (False, True):
        es = sorted([e for e in r if bool(e[3] & 2) == bridge], key=rank)
        for k, lanes, ln, fl in es:
            w = width(k, lanes)
            if k in DRIVE[2:] or fl & 4:
                lay += [("walk", ln.buffer(w / 2 + 2.6))] * bool(fl & 4) + [("curb", ln.buffer(w / 2 + 0.3))]
        for k, lanes, ln, fl in es:
            w = width(k, lanes)
            if k in ("rail", "light_rail", "subway", "tram"):
                lay += [("ballast", ln.buffer(1.8))] + [("rail", ln.offset_curve(s * 0.72).buffer(0.06)) for s in (-1, 1)]
            else:
                lay.append(({"footway": "path", "cycleway": "path", "track": "sand"}.get(k, "asphalt"), ln.buffer(w / 2)))
        lay += [("line", m.difference(cut)) for e in es for m in markings(*e)]
        if not bridge:
            lay += [("line", shapely.unary_union(zebra))] + [("building", q) for q in b if q.intersects(clip)]
    return lay


def raster(lay, x0, y0, n, ss=2):
    m = n * ss
    img, cls = Image.new("RGB", (m, m)), Image.new("L", (m, m))
    di, dc = ImageDraw.Draw(img), ImageDraw.Draw(cls)
    T = lambda cs: [((x - x0) / E * m, (y0 + E - y) / E * m) for x, y in cs]
    view = box(x0, y0, x0 + E, y0 + E).buffer(1)
    for name, geom in lay:
        geom = geom.intersection(view)
        for q in getattr(geom, "geoms", [geom]):
            if q.geom_type != "Polygon" or q.is_empty:
                continue
            ring, fill = T(q.exterior.coords), PAL[name]
            if q.interiors:
                mask = Image.new("L", (m, m))
                [ImageDraw.Draw(mask).polygon(r, fill=v) for r, v in [(ring, 255)] + [(T(h.coords), 0) for h in q.interiors]]
                img.paste(fill, mask=mask), cls.paste(CLS[name], mask=mask)
            else:
                di.polygon(ring, fill=fill, outline=(70, 70, 70) if name == "building" else None, width=ss * 2)
                dc.polygon(ring, fill=CLS[name])
    return img.resize((n, n), Image.LANCZOS), cls.resize((n, n), Image.NEAREST)


def ortho(name, x0, y0):
    f = RAW / f"ref/{name}.jpg"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        e, n = x0 + 370000, y0 + 6670000
        r = requests.get(WMS, params={"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": "Ortokartta", "STYLES": "",
                                      "SRS": "EPSG:3067", "FORMAT": "image/jpeg", "WIDTH": 512, "HEIGHT": 512,
                                      "BBOX": f"{e},{n},{e + E},{n + E}"}, timeout=300)
        Image.open(io.BytesIO(r.content)).convert("RGB").resize((1024, 1024), Image.LANCZOS).save(f, quality=90)
    return f


def body(name):
    url = lambda f, t: f"data:image/{t};base64," + base64.b64encode(f.read_bytes()).decode()
    return {"model": MODEL, "prompt": PROMPT.format(e=E), **OPT,
            "images": [{"image_url": url(RAW / f"{name}_ctl.png", "png")}, {"image_url": url(RAW / f"ref/{name}.jpg", "jpeg")}]}


def api(method, path, **kw):
    key = Path.home().joinpath(".secrets/agent-secrets/openai.txt").read_text().strip()
    return requests.request(method, "https://api.openai.com/v1/" + path, headers={"Authorization": f"Bearer {key}"}, timeout=900, **kw)


def save(name, r):
    with open(RAW / "usage.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "usage": r.get("usage"), "error": r.get("error")}) + "\n")
    if "data" not in r:
        return None
    Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"]))).convert("RGB").resize((N, N), Image.LANCZOS).save(
        RAW / f"{name}.jpg", quality=93)
    return check(name)


def check(name):
    g = ndimage.gaussian_filter(np.asarray(Image.open(RAW / f"{name}.jpg").convert("L"), float), 0.7)
    cls = np.asarray(Image.open(RAW / f"{name}_cls.png"))
    m = cls == CLS["line"]
    d = ndimage.distance_transform_edt(~((g > 175) & ndimage.binary_dilation(m | (cls == CLS["asphalt"]), iterations=4)))[m] * E / N
    e = [round(float(np.percentile(d, q)), 2) if len(d) else 0.0 for q in (50, 90)]
    with open(RAW / "align.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "err": e}) + "\n")
    if e[1] > 0.3:
        (RAW / f"{name}.jpg").rename(RAW / f"{name}_rej{e[1]:.2f}_{time.time():.0f}.jpg")
    return e


def prep(ij):
    i, j = ij
    name, x0, y0 = f"{i}_{j}", S * i - S * PAD, S * j - S * PAD
    if not (RAW / f"{name}_cls.png").exists():
        ctl, cls = raster(scene(x0, y0), x0, y0, N)
        ctl.save(RAW / f"{name}_ctl.png")
        cls.save(RAW / f"{name}_cls.png")
    ortho(name, x0, y0)
    return name


def spent():
    rows = [json.loads(l) for l in (RAW / "usage.jsonl").read_text().splitlines()]
    return sum((u["input_tokens_details"]["text_tokens"] * 5 + u["input_tokens_details"].get("image_tokens", 0) * 8 + u["output_tokens"] * 30)
               / 1e6 for r in rows if (u := r["usage"]))


def gen(ij):
    name = prep(ij)
    if (np.asarray(Image.open(RAW / f"{name}_cls.png")) == CLS["water"]).mean() > 0.999:
        return
    if len(rej := sorted(RAW.glob(f"{name}_rej*.jpg"))) >= 3 and not (RAW / f"{name}.jpg").exists():
        return rej[0].rename(RAW / f"{name}.jpg")
    for _ in range(3):
        if (RAW / f"{name}.jpg").exists() or spent() > CAP:
            return
        r = api("POST", "images/edits", json=body(name)).json()
        print(name, save(name, r) if "data" in r else r.get("error"), flush=True)


def area(radius):
    r, c = float(radius), np.floor(C / S).astype(int)
    k = int(r // S) + 1
    return [(c[0] + a, c[1] + b) for a in range(-k, k + 1) for b in range(-k, k + 1)
            if np.hypot(*(np.array([c[0] + a + 0.5, c[1] + b + 0.5]) * S - C)) < r + S * 0.71]


def run(radius, size="2048", quality="high"):
    OPT.update(size=f"{size}x{size}", quality=quality)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(gen, sorted(area(radius), key=lambda t: np.hypot(*(np.array(t) + 0.5) * S - C))))
    stitch()


def stitch():
    tiles = sorted(tuple(map(int, f.stem.split("_"))) for f in RAW.glob("*.jpg") if f.stem.count("_") == 1)
    K = len(PAL)

    def stat(t):
        a, c = np.asarray(Image.open(RAW / f"{t[0]}_{t[1]}.jpg"), np.float64), np.asarray(Image.open(RAW / f"{t[0]}_{t[1]}_cls.png")).ravel()
        return np.stack([np.bincount(c, a[..., q].ravel(), K) for q in range(3)], -1), np.bincount(c, None, K)[:, None]

    with ThreadPoolExecutor(8) as ex:
        st = dict(zip(tiles, ex.map(stat, tiles)))
    glob = sum(v[0] for v in st.values()) / (sum(v[1] for v in st.values()) + 1e-6)
    cache = {}

    def load(t):
        if t not in cache:
            tot, cnt = st[t]
            gain = np.where(cnt > 2000, glob / np.maximum(tot / (cnt + 1e-6), 1), 1).clip(0.75, 1.3)
            full = Image.open(RAW / f"{t[0]}_{t[1]}_cls.png")
            cls = np.asarray(full.resize((N // 8, N // 8), Image.NEAREST))
            g = ndimage.zoom(ndimage.gaussian_filter(gain[cls], (5, 5, 0)), (8, 8, 1), order=1)
            land = ndimage.gaussian_filter((np.asarray(full) != CLS["water"]) * 255.0, 0.4 * N / E)
            rgb = (np.asarray(Image.open(RAW / f"{t[0]}_{t[1]}.jpg"), np.float32) * g).clip(0, 255)
            cache[t] = np.dstack([rgb, land]).astype(np.uint8)
        return cache[t]

    ramp = np.clip((np.arange(N) + 0.5) * E / N / (2 * S * PAD), 0, 1)
    w = np.outer(np.minimum(ramp, ramp[::-1]), np.minimum(ramp, ramp[::-1]))[..., None] + 1e-4
    o, step, n = round(S * PAD / E * N), S / E * N, round(S / PX)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, j in tiles:
        [cache.pop(t) for t in list(cache) if t[0] < i - 1]
        acc, ws = np.zeros((N - 2 * o, N - 2 * o, 4), np.float32), np.zeros((N - 2 * o, N - 2 * o, 1), np.float32)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if (i + di, j + dj) not in st:
                    continue
                dx, dy = round(di * step), round(-dj * step)
                ys, xs = slice(max(o, dy), min(N - o, N + dy)), slice(max(o, dx), min(N - o, N + dx))
                src = np.s_[ys.start - dy:ys.stop - dy, xs.start - dx:xs.stop - dx]
                dst = np.s_[ys.start - o:ys.stop - o, xs.start - o:xs.stop - o]
                acc[dst] += load((i + di, j + dj))[src] * w[src]
                ws[dst] += w[src]
        img, f = (acc / ws).clip(0, 255).astype(np.uint8), OUT / f"{i}_{j}.webp"
        f.unlink(missing_ok=True)
        if img[..., 3].max() > 8:
            Image.fromarray(img).resize((n, n), Image.LANCZOS).save(f, quality=70, method=4)
    [f.unlink() for f in OUT.glob("*.webp") if tuple(map(int, f.stem.split("_"))) not in st]
    tiles = sorted(tuple(map(int, f.stem.split("_"))) for f in OUT.glob("*.webp"))
    (VIZ / "ground.json").write_text(json.dumps({"px": PX, "size": S, "tiles": [list(t) for t in tiles]}))


SPRITES = {"birch": ("silver birch trees (Betula pendula) with white bark", [15, 13, 17]),
           "pine": ("Scots pine trees (Pinus sylvestris) with tall orange-brown upper trunks and flat airy crowns", [18, 16, 20]),
           "spruce": ("Norway spruce trees (Picea abies), dense dark conical", [20, 16, 12]),
           "linden": ("broadleaf street trees, small-leaved lime and Norway maple, rounded full crowns", [14, 12, 10]),
           "shrub": ("low garden shrubs and bushes of different kinds, wider than tall", [1.5, 1.2, 1.8, 1.0]),
           "walker": ("pedestrians walking in different directions in plain unbranded everyday Finnish autumn clothes, full body", [1.75, 1.65, 1.8, 1.7, 1.6, 1.85])}


def sprites():
    d = ROOT / "viz/assets/sprites"
    d.mkdir(parents=True, exist_ok=True)
    key, meta = Path.home().joinpath(".secrets/agent-secrets/openai.txt").read_text().strip(), []
    for name, (what, hs) in SPRITES.items():
        if not (RAW / f"sprite_{name}.png").exists():
            r = requests.post("https://api.openai.com/v1/images/generations", headers={"Authorization": f"Bearer {key}"}, json={
                "model": MODEL, "size": "1536x1024", "quality": "high", "background": "transparent", "output_format": "png",
                "prompt": f"{len(hs)} separate {what}, typical of Espoo, Finland, in one row with wide empty space between them, each "
                "fully visible, seen from an elevated viewpoint looking down at about 45 degrees as in an isometric city illustration, "
                "so the top of each crown or head is partly visible. Painterly, finely detailed architectural illustration style, "
                "soft even overcast light, no cast shadow, no ground, no base, no text, no logos, isolated on a fully transparent background."}, timeout=900).json()
            with open(RAW / "usage.jsonl", "a") as f:
                f.write(json.dumps({"name": name, "usage": r.get("usage"), "error": r.get("error")}) + "\n")
            Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"]))).save(RAW / f"sprite_{name}.png")
        im = np.asarray(Image.open(RAW / f"sprite_{name}.png").convert("RGBA"), float)
        lab, n = ndimage.label(ndimage.binary_dilation(im[..., 3] > 24, iterations=6))
        objs = sorted(ndimage.find_objects(lab), key=lambda s: -(s[0].stop - s[0].start) * (s[1].stop - s[1].start))[:len(hs)]
        for k, (sl, h) in enumerate(zip(sorted(objs, key=lambda s: s[1].start), hs)):
            a = np.clip((im[sl][..., 3] - 40) / 175, 0, 1) * (lab[sl] == lab[sl][lab[sl] > 0].max())
            solid = a > 0.99
            idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
            rgb = im[sl][..., :3][idx[0], idx[1]]
            ys, xs = np.nonzero(a > 0.5)
            base = [float(xs[ys > ys.max() - 6].mean() / a.shape[1]), 1.0]
            top = ys < ys.min() + 0.6 * (ys.max() - ys.min())
            crown = [float(xs[top].mean() / a.shape[1]), float(ys[top].mean() / a.shape[0])]
            out = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8))
            out = out.crop((0, 0, a.shape[1], int(ys.max()) + 1))
            s = min(1, (128 if name == "walker" else 384) / out.height)
            out = out.resize((round(out.width * s), round(out.height * s)), Image.LANCZOS)
            out.save(d / f"{name}{k}.png", optimize=True)
            meta.append({"name": f"{name}{k}", "w": out.width, "h": out.height, "m": h, "base": base,
                         "crown": crown if name not in ("walker", "shrub") else base})
    (d / "sprites.json").write_text(json.dumps(meta, indent=0))


BANK = {"grass": (["grass"], 20, 512, 2), "forest_floor": (["forest"], 24, 512, None), "asphalt": (["asphalt"], 10, 256, None),
        "gravel": (["land", "sand"], 16, 256, "dry"), "meadow": (["land"], 24, 512, "green"), "sand": (["sand"], 16, 256, None),
        "rock": (["rock"], 16, 256, None)}


def windows(classes, m, rule, pure=0.999):
    w, out = round(m / (E / N) / 8 * (1.45 if rule == 2 else 1)), []
    for t in area(600):
        f = RAW / f"{t[0]}_{t[1]}.jpg"
        if not f.exists():
            continue
        c = np.asarray(Image.open(RAW / f"{t[0]}_{t[1]}_cls.png"))
        good = np.isin(c, [CLS[k] for k in classes]) & ~ndimage.binary_dilation(c == CLS["line"], iterations=9)
        ok = ndimage.uniform_filter(good.reshape(N // 8, 8, N // 8, 8).mean((1, 3)), w, mode="constant") >= pure
        ok[:w // 2 + 2], ok[-w // 2 - 2:], ok[:, :w // 2 + 2], ok[:, -w // 2 - 2:] = False, False, False, False
        for y, x in np.argwhere(ok)[::max(1, int(ok.sum()) // 6)]:
            out.append((f, (x - w // 2) * 8, (y - w // 2) * 8, w * 8))
    return out if len(out) >= 12 or pure < 0.955 else windows(classes, m, rule, pure - 0.01)


def pick(cands, k, rule):
    feats, crops = [], []
    for f, x, y, w in cands:
        im = np.asarray(Image.open(f).crop((x, y, x + w, y + w)).resize((64, 64)), float)
        g = im[..., 1] - im[..., [0, 2]].mean(-1)
        if rule == "dry" and g.mean() > 4 or rule == "green" and g.mean() < 10:
            continue
        hi = np.asarray(Image.open(f).crop((x, y, x + w, y + w)).convert("L"), float)
        spec = np.abs(np.fft.fft2(hi - hi.mean()))[2:-2, 2:-2]
        if spec.max() > 40 * spec.mean():
            continue
        feats.append(np.r_[im.mean((0, 1)), 3 * im.std(), 2 * g.mean(), (hi - ndimage.gaussian_filter(hi, 2)).std()])
        crops.append((f, x, y, w))
    if not crops:
        return []
    feats = np.array(feats)
    keep = feats[:, -1] >= 0.75 * np.median(feats[:, -1])
    feats, crops = feats[keep][:, :-1], [c for c, k in zip(crops, keep) if k]
    chosen = []
    for i in np.argsort(np.linalg.norm((feats - np.median(feats, 0)) / (feats.std(0) + 1e-6), axis=1)):
        if len(chosen) < k and all(crops[i][0] != crops[c][0] or max(abs(crops[i][1] - crops[c][1]), abs(crops[i][2] - crops[c][2])) > crops[i][3] for c in chosen):
            chosen.append(i)
    return [crops[c] for c in chosen]


def despeck(im):
    a = np.asarray(im, np.float32)
    for _ in range(6):
        l = a.mean(-1)
        m = ndimage.binary_dilation((l > l.mean() + 2.6 * l.std()) | (l < l.mean() - 3 * l.std()), iterations=4)
        if not m.any():
            break
        for _ in range(6):
            a[m] = np.roll(a, (37, 23), (0, 1))[m]
            m &= np.roll(m, (37, 23), (0, 1))
    return Image.fromarray(a.astype(np.uint8))


def noise(n, sigma, seed):
    z = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=n), sigma, mode="wrap")
    return (z - z.mean()) / z.std()


def bricks(n, ppm, w, h, bond, col, grain, seed, moss=0):
    y, x = np.mgrid[0:n[0], 0:n[1]] / ppm
    row = np.floor(y / h).astype(int)
    xs = x + (row % 2) * w / 2 * bond
    col_i = np.floor(xs / w).astype(int)
    u, v = (xs % w) * ppm, (y % h) * ppm
    e = np.minimum.reduce([u, w * ppm - u, v, h * ppm - v])
    jit = np.random.default_rng(seed).normal(0, 0.035, (row.max() + 1, col_i.max() + 1))[row, col_i]
    lum = (1 + jit + grain[0] * noise(n, 0.7, seed) + grain[1] * noise(n, 3, seed + 1)) * np.clip(0.55 + e / 5, 0, 1)
    out = np.array(col, float) * lum[..., None]
    if moss:
        g = np.clip(1.2 - e / 2.5, 0, 1) * np.clip(noise(n, 12, seed + 2) + 0.3, 0, 1) * moss
        out = out * (1 - g[..., None]) + np.array([72, 88, 52]) * g[..., None]
    return out


def stripes(a, ppm):
    m = np.zeros(a.shape[:2])
    y, x = np.mgrid[0:a.shape[0], 0:a.shape[1]] / ppm
    m[(np.abs((x - 1.25) % 2.5 - 1.25) < 0.06) & (((y - 2) % 20) < 10)] = 1
    m[np.abs((y - 7) % 20) < 0.06] = 1
    m = ndimage.gaussian_filter(m, 0.6, mode="wrap") * np.clip(0.85 + 0.12 * noise(m.shape, 2, 5), 0, 1) * (noise(m.shape, 3, 6) > -2.2)
    return a * (1 - m[..., None]) + np.array([228, 228, 222]) * m[..., None]


def extras(d):
    hp = []
    for f, x, y, w in windows(["walk", "paving"], 6, None)[:40]:
        a = np.asarray(Image.open(f).crop((x, y, x + w, y + w)), float)
        hp.append(np.r_[a.mean((0, 1)), (a.mean(-1) - ndimage.gaussian_filter(a.mean(-1), 1.5)).std() / a.mean()])
    st = np.median(hp, 0)
    base, gr = st[:3], (0.6 * st[3], 0.5 * st[3])
    warm = base * np.array([1.025, 1.0, 0.965])
    yield "paving", 6, [bricks((512, 512), 512 / 6, w, h, bond, c, gr, k, moss) for k, (w, h, bond, c, moss) in enumerate(
        [(0.4, 0.25, 1, base * 1.03, 0), (0.4, 0.25, 1, warm, 0), (0.4, 0.25, 1, base * 0.97, 0.35), (0.3, 0.3, 0, base, 0)])]
    walk = []
    for k, c in enumerate((base * 1.08, warm)):
        a = bricks((512, 160), 64, 0.25, 0.4, 1, c, gr, 10 + k)
        curb = bricks((512, 19), 64, 0.3, 1.0, 0, (205, 203, 197), (gr[0] * 1.5, gr[1]), 20 + k)
        curb[:, :3] *= 0.75
        a[:, :19] = curb
        walk.append(a)
    yield "sidewalk", 8, walk
    yield "parking", 20, [stripes(np.tile(np.asarray(Image.open(f).convert("RGB"), float), (2, 2, 1)), 512 / 20) for f in sorted(d.glob("asphalt-*.webp"))]


def put(d, meta, name, m, ims, norm=True):
    for f in [*d.glob(f"{name}-*.webp"), d / f"{name}.webp"]:
        f.unlink(missing_ok=True)
    mu, sd = np.mean([a.mean((0, 1)) for a in ims], 0), np.mean([a.std((0, 1)) for a in ims], 0)
    for k, a in enumerate(ims):
        a = ((a - a.mean((0, 1))) * np.clip(sd / (a.std((0, 1)) + 1e-6), 0.8, 1.25) + mu) if norm else a
        Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(d / f"{name}-{k}.webp", quality=88, method=6)
    meta[name] = {"name": name, "metres": m, "variants": len(ims)}
    print(name, len(ims), flush=True)


def orient(im, size):
    g = ndimage.gaussian_filter(np.asarray(im.convert("L"), float), 1.5)
    gy, gx = ndimage.sobel(g, 0), ndimage.sobel(g, 1)
    phi, w = np.arctan2(gy, gx), gx ** 2 + gy ** 2
    im = im.rotate(np.degrees(np.arctan2((w * np.sin(2 * phi)).sum(), (w * np.cos(2 * phi)).sum()) / 2) - 90, Image.BICUBIC)
    c = (im.size[0] - size) // 2
    return im.crop((c, c, c + size, c + size))


def seamless(img):
    b = np.asarray(img, np.float32)
    n = len(b)
    a, mu = np.roll(b, (n // 2, n // 2), (0, 1)), b.mean((0, 1))
    g = b.mean(-1) - ndimage.gaussian_filter(b.mean(-1), n / 128, mode="wrap")
    c = np.fft.ifft2(np.fft.fft2(np.roll(g, (n // 2, n // 2), (0, 1))) * np.conj(np.fft.fft2(g))).real
    r = np.minimum(np.arange(n), n - np.arange(n))
    c[(np.maximum.outer(r, r) > 0.3 * n)] = -np.inf
    b = np.roll(b, np.unravel_index(c.argmax(), c.shape), (0, 1))
    d = np.abs(np.arange(n) - n / 2 + 0.5)
    w = np.clip((0.2 * n - np.minimum.outer(d, d)) / (0.12 * n), 0, 1)[..., None]
    t = mu + ((1 - w) * (a - mu) + w * (b - mu)) / np.sqrt((1 - w) ** 2 + w ** 2)
    low = ndimage.gaussian_filter(t, (n / 4, n / 4, 0), mode="wrap")
    return Image.fromarray((t - low + low.mean((0, 1))).clip(0, 255).astype(np.uint8))


def worn(alpha, seed):
    n = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=alpha.shape), 2, mode="wrap")
    n /= n.std()
    return alpha * (n > -2.1) * np.clip(0.88 + 0.08 * n, 0, 1)


def marks(d, P=64, k=4):
    atlas, out = Image.new("L", (1024 * k, 512 * k)), {}
    for name, x0, w, h, xs in (("zebra", 16, 2.5, 4, (-1.25, -0.25, 0.75)), ("stop", 200, 3.5, 0.5, (-1.75,))):
        for x in xs:
            ImageDraw.Draw(atlas).rectangle([(x0 + (x + w / 2) * P) * k, 16 * k, (x0 + (x + w / 2 + (0.5 if name == "zebra" else w)) * P) * k, (16 + h * P) * k], fill=255)
        out[name] = {"uv": [x0 / 1024, 16 / 512, (x0 + w * P) / 1024, (16 + h * P) / 512], "metres": [w, h]}
    a = worn(np.asarray(atlas.resize((1024, 512), Image.LANCZOS), float) / 255, 2)
    Image.fromarray(np.dstack([np.full((512, 1024, 3), 238), a * 255]).astype(np.uint8), "RGBA").save(d / "marks.webp", lossless=True)
    return out


def bank(*only):
    d = ASSETS
    meta = {m["name"]: m for m in json.loads((d / "materials.json").read_text())["materials"]}
    for name, (classes, m, px, rule) in BANK.items():
        if only and name not in only:
            continue
        ims = []
        for f, x, y, w in pick(windows(classes, m, rule), 4, rule):
            im = Image.open(f).crop((x, y, x + w, y + w))
            im = despeck(orient(im, round(w / 1.45)) if rule == 2 else im)
            ims.append(np.asarray(despeck(seamless(im.resize((px, px), Image.LANCZOS))), float))
        ims += [np.rot90(a, 2 if rule == 2 else 1).copy() for a in ims][:max(0, 4 - len(ims))]
        put(d, meta, name, m, ims)
    for name, m, ims in extras(d):
        if not only or name in only:
            put(d, meta, name, m, ims, name != "paving")
    (d / "materials.json").write_text(json.dumps({"materials": list(meta.values()), "marks": marks(d)}))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
