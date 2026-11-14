import base64, gzip, io, json, re, struct, sys, time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
import shapely
from PIL import Image, ImageEnhance, ImageFilter
from pyproj import Transformer
from shapely import constrained_delaunay_triangles
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from data.geo import ROOT, VIZ
from data.hype import HYPE_NW, HYPE_SE, HYPE_SW, PARKING

RAW, TEX, OUT, DET = ROOT / "data/raw/district", ROOT / "data/district", ROOT / "viz/assets", VIZ / "detail"
WFS = "https://kartat.espoo.fi/teklaogcweb/wfs.ashx"
WMS = "https://kartat.espoo.fi/teklaogcweb/wms.ashx"
C, E0 = np.array([-1511.7, 1239.2]), np.array([370000, 6670000])
NEAR, FAR, TS = 600, 1500, 1000
OWN = unary_union([Polygon([HYPE_SW, HYPE_SE, (HYPE_SE[0] + HYPE_NW[0] - HYPE_SW[0], HYPE_SE[1] + HYPE_NW[1] - HYPE_SW[1]), HYPE_NW]), Polygon(PARKING)]).buffer(3)
DIST = (C - 1560, 0.3, 10400)
GK, KG = Transformer.from_crs(3879, 3067, always_xy=True), Transformer.from_crs(3067, 3879, always_xy=True)
NS = {"gml": "http://www.opengis.net/gml", "bldg": "http://www.opengis.net/citygml/building/2.0", "gen": "http://www.opengis.net/citygml/generics/2.0"}
BG = (205, 214, 224)
Image.MAX_IMAGE_PIXELS = None


def get(url, ok, **params):
    for k in range(10):
        try:
            r = requests.get(url, params=params, timeout=600)
            if r.ok and ok(r.content):
                return r.content
        except requests.RequestException:
            pass
        time.sleep(5 * 2 ** min(k, 5))
    raise RuntimeError(f"{url} {params}")


def save(path, data):
    (tmp := path.with_suffix(".part")).write_bytes(data)
    tmp.rename(path)


def read(path):
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def wfs(path, x0, y0, x1, y1):
    for name, typ, ver in (("lod2", "bldg:building_lod2", "1.1.0"), ("kanta", "kanta:Rakennus", "1.0.0")):
        if not (p := path(name)).exists():
            a, b = KG.transform(*np.array([[x0, x1, x0, x1], [y0, y0, y1, y1]]) + E0[:, None])
            data = get(WFS, lambda c: b"FeatureCollection" in c[:3000], SERVICE="WFS", VERSION=ver, REQUEST="GetFeature",
                       TYPENAME=typ, SRSNAME="EPSG:3879", BBOX=f"{min(a):.0f},{min(b):.0f},{max(a):.0f},{max(b):.0f}")
            save(p, gzip.compress(data) if p.suffix == ".gz" else data)


def ortho(path, o, px, n):
    if not path.exists():
        k = -(-n // 2080)
        img, step = Image.new("RGB", (n, n)), n // k
        for i in range(k):
            for j in range(k):
                e, s = o + E0 + (i * step * px, (n - (j + 1) * step) * px)
                img.paste(Image.open(io.BytesIO(get(WMS, lambda c: c[:1] != b"<", SERVICE="WMS", VERSION="1.1.1", REQUEST="GetMap", LAYERS="Ortokartta",
                                                     STYLES="", SRS="EPSG:3067", FORMAT="image/jpeg", WIDTH=step, HEIGHT=step,
                                                     BBOX=f"{e},{s},{e + step * px},{s + step * px}"))).convert("RGB"), (i * step, j * step))
        img.save(buf := io.BytesIO(), "JPEG", quality=90)
        save(path, buf.getvalue())
    return Image.open(path), (o, px, n)


def parse(lod, kanta):
    mats = dict(re.findall(rb"<kanta:rakennustunnus>(\w+)</kanta:rakennustunnus>.*?<kanta:julkisivumateriaali>([^<]*)<", kanta, flags=re.S))
    out = []
    for _, el in ET.iterparse(io.BytesIO(lod)):
        if el.tag != f"{{{NS['bldg']}}}Building":
            continue
        a = {g.get("name"): g.findtext("gen:value", namespaces=NS) for g in el.findall("gen:stringAttribute", NS)}
        faces = []
        for s in (s for bb in el.findall("bldg:boundedBy", NS) for s in bb):
            for p in s.iter(f"{{{NS['gml']}}}Polygon"):
                rings = []
                for c in p.iter(f"{{{NS['gml']}}}LinearRing"):
                    v = np.array(" ".join(e.text for e in c).split(), float).reshape(-1, 3)
                    x, y = GK.transform(v[:, 0], v[:, 1])
                    rings.append(np.c_[x - E0[0], y - E0[1], v[:, 2]])
                faces.append((s.tag.split("}")[1][0], rings))
        h = float(el.findtext("bldg:measuredHeight", "0", NS))
        el.clear()
        if faces:
            i = a.get("rakennustunnus") or "x%.0f_%.0f" % tuple(faces[0][1][0][0, :2])
            out.append({"id": i, "use": a.get("kayttotarkoitus") or "", "year": int((a.get("valmistumispvm") or "0")[:4]),
                        "floors": int(a.get("kerrosluku") or 0), "mat": mats.get(i.encode(), b"").decode(), "faces": faces, "h": h})
    return out


GROUPS = [("house", "Omakoti|Pari|Sauna|vapaa-ajan"), ("row", "Rivi"), ("apt", "kerrosta|Erityis|Asuntola"),
          ("shop", "Kauppa|myymälä|Ravintola|majoitus"), ("office", "Toimisto"), ("parking", "Pysäköinti"),
          ("public", "päiväkod|oppilai|Urheilu|urheilu|kokoontumis|Seura|Uskonnon|Laitos|hoito"),
          ("industry", "teollisuus|Lämpimät"), ("transit", "liikenteen")]
BKIND = {"residential": "house", "apartments": "apt", "commercial": "shop", "industrial": "industry", "public": "public",
         "garage": "parking", "other": "shed"}
MAT = {"puu": "wood", "tiili": "brick", "betoni": "concrete", "metallilevy": "metal"}


def style(b):
    g = b["use"] if b["use"] in BKIND.values() else next((g for g, p in GROUPS if re.search(p, b["use"])),
                                                         "apt" if b["floors"] >= 3 else "industry" if not b["use"] and b["h"] > 7 else "shed")
    return g, "old" if b["year"] < 1990 else "mid" if b["year"] < 2010 else "new", MAT.get(b["mat"], "render")


def normals(r):
    n = np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]) if len(r) else np.zeros((0, 3))
    a = np.linalg.norm(n, axis=1) + 1e-9
    return n / a[:, None], a


def triangulate(rings):
    r0 = rings[0]
    n = np.cross(r0, np.roll(r0, -1, 0)).sum(0)
    if np.linalg.norm(n) < 1e-6:
        return np.zeros((0, 3, 3)), np.zeros(3)
    n = n / np.linalg.norm(n)
    a = np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    a /= np.linalg.norm(a)
    b, o = np.cross(n, a), r0[0]
    rings = [np.c_[(r - o) @ a, (r - o) @ b] for r in rings if len(np.unique(r.round(3), axis=0)) > 2]
    p = Polygon(rings[0], rings[1:]) if rings and len(rings[0]) > 2 else Polygon()
    q = shapely.get_coordinates(constrained_delaunay_triangles(p if p.is_valid else p.buffer(0))).reshape(-1, 4, 2)[:, :3]
    v = o + q[..., :1] * a + q[..., 1:] * b
    flip = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]) @ n <= 0
    v[flip] = v[flip][:, ::-1]
    return v, n


def prism(ring, h, gable=False):
    ring = np.asarray(ring)[:-1] if np.allclose(ring[0], ring[-1]) else np.asarray(ring)
    ring = ring if Polygon(ring).exterior.is_ccw else ring[::-1]
    r = np.array(Polygon(ring).minimum_rotated_rectangle.exterior.coords)[:-1]
    if not (gable and len(r) == 4 and 30 < Polygon(ring).area < 400 and Polygon(ring).area > 0.8 * Polygon(r).area):
        return [("W", [np.array([(*a, 0), (*b, 0), (*b, h), (*a, h)])]) for a, b in zip(ring, np.roll(ring, -1, 0))] + [("R", [np.c_[ring, np.full(len(ring), h)]])]
    r = r if Polygon(r).exterior.is_ccw else r[::-1]
    r = r if np.linalg.norm(r[1] - r[0]) >= np.linalg.norm(r[2] - r[1]) else np.roll(r, -1, 0)
    e = h - min(np.linalg.norm(r[2] - r[1]) * 0.3, h * 0.4)
    lo, up, m0, m1 = np.c_[r, 0 * r[:, 0]], np.c_[r, 0 * r[:, 0] + e], (*(r[0] + r[3]) / 2, h), (*(r[1] + r[2]) / 2, h)
    return [("W", [np.array(f)]) for f in ([lo[0], lo[1], up[1], up[0]], [lo[2], lo[3], up[3], up[2]], [lo[1], lo[2], up[2], m1, up[1]],
                                            [lo[3], lo[0], up[0], m0, up[3]])] + [("R", [np.array(f)]) for f in ([up[0], up[1], m1, m0], [up[2], up[3], m0, m1])]


def residual(foot, tiles, keep):
    taken, out, rings, kinds = unary_union(foot), [], [], json.loads((VIZ / "tiles.json").read_text())["bkinds"]
    for i, j in tiles:
        for h, k, xy in json.loads((VIZ / f"tiles/{i}_{j}.json").read_text())["b"]:
            ring = np.array(xy).reshape(-1, 2)
            if not keep(ring.mean(0)):
                continue
            p = Polygon(ring).buffer(0)
            inner = p.contains(Point(ring.mean(0)))
            rings.append(list(np.round(ring if inner else ring.mean(0) + [(-1, -1), (1, -1), (1, 1), (-1, 1)], 1).ravel()))
            if p.area > 4 and p.intersection(taken).area < 0.3 * p.area:
                out.append({"id": f"t{ring[0][0]:.0f}_{ring[0][1]:.0f}", "use": BKIND[kinds[k]], "year": 2020, "floors": round(h / 3),
                            "mat": "", "faces": prism(ring, h, kinds[k] == "residential"), "foot": p, "h": h})
    return out, rings


def prep(b):
    g = [r[0] for k, r in b["faces"] if k == "G"]
    pts = np.vstack([r[0] for _, r in b["faces"]])
    base = min(r[:, 2].min() for r in g) if g else 0
    b.setdefault("foot", unary_union([Polygon(r[:, :2]).buffer(0) for r in g]) if g else Polygon())
    faces = [(k, [r - (0, 0, base) for r in rs]) for k, rs in b["faces"] if k != "G"]
    if pts[:, 2].max() - base < 1:
        h, big = b["h"] if b["h"] > 1 else b["floors"] * 3.2, max(getattr(b["foot"], "geoms", [b["foot"]]), key=lambda p: p.area)
        faces = prism(big.exterior.coords, h) if h > 1 and big.area > 0 else []
    mrr = np.array((b["foot"] if b["foot"].area > 1 else Polygon(pts[:, :2]).convex_hull).minimum_rotated_rectangle.exterior.coords)
    eu = (eu := max((mrr[1] - mrr[0], mrr[2] - mrr[1]), key=np.linalg.norm)) / (np.linalg.norm(eu) or 1)
    axes = [eu, np.array([-eu[1], eu[0]]), -eu, np.array([eu[1], -eu[0]])]
    tops = [(Polygon(r[0][:, :2]).buffer(0), r[0][:, 2].max()) for k, r in faces if k == "R" and len(r[0]) > 2]
    sides, roof = [{"e": axes[(k + 1) % 4], "tris": []} for k in range(4)], []
    for k, rs in faces:
        rs = [np.c_[r[:, :2], np.maximum(r[:, 2], 0)] for r in rs]
        t, n = triangulate(rs)
        if k != "W" and abs(n[2]) > 0.3:
            roof.append(t if n[2] > 0 else t[:, ::-1])
        elif len(t):
            q = Point(rs[0][:, :2].mean(0) + 0.4 * n[:2] / (np.linalg.norm(n[:2]) or 1))
            if any(z > rs[0][:, 2].mean() and p.contains(q) for p, z in tops):
                t, n = t[:, ::-1], -n
            sides[int(np.argmax([n[:2] @ x for x in axes]))]["tris"].append(t)
    b["walls"] = [w for w in sides if w["tris"]]
    for w in b["walls"]:
        w["tris"] = np.vstack(w["tris"])
        u = w["tris"][..., :2] @ w["e"]
        w["u0"], w["u1"], w["z0"] = u.min(), u.max(), w["tris"][..., 2].min()
        w["zt"] = max(w["tris"][..., 2].max(), w["z0"] + 0.5)
    b["roof"] = np.vstack(roof) if roof else np.zeros((0, 3, 3))
    b["c"] = np.array(b["foot"].centroid.coords[0]) if not b["foot"].is_empty else pts[:, :2].mean(0)
    b["d"], b["style"] = float(np.hypot(*(b["c"] - C))), style(b)
    b["fh"] = float(np.clip(b["h"] / b["floors"], 2.7, 4.5)) if b["floors"] > 0 and b["h"] > 0 else 3.0
    return b


def buildings():
    wfs(lambda name: RAW / f"{name}.xml", *(DIST[0] - 50), *(DIST[0] + 3170))
    bs = [prep(b) for b in parse(read(RAW / "lod2.xml"), read(RAW / "kanta.xml")) if np.hypot(*(b["faces"][0][1][0][:, :2].mean(0) - C)) < FAR + 100]
    bs = [b for b in bs if b["d"] < FAR and (len(b["roof"]) or b["walls"])]
    near = [t for t in json.loads((VIZ / "tiles.json").read_text())["tiles"] if np.hypot((t[0] + 0.5) * TS - C[0], (t[1] + 0.5) * TS - C[1]) < FAR + 800]
    extra, rings = residual([b["foot"] for b in bs], near, lambda c: np.hypot(*(c - C)) < FAR)
    return [b for b in bs + [prep(b) for b in extra] if b["foot"].is_empty or not OWN.contains(b["foot"].centroid)], rings


def pack(sizes, w, h, pad=2, pages=99):
    pos, page, x, y, shelf = [None] * len(sizes), 0, 0, 0, 0
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i][1]):
        sw, sh = sizes[i][0] + 2 * pad, sizes[i][1] + 2 * pad
        if x + sw > w:
            x, y, shelf = 0, y + shelf, 0
        if y + sh > h:
            page, x, y, shelf = page + 1, 0, 0, 0
        if page >= pages or sw > w or sh > h:
            return None
        pos[i] = (page, x + pad, y + pad)
        x, shelf = x + sw, max(shelf, sh)
    return pos


def size(f, s):
    return int(np.ceil((f["u1"] - f["u0"]) * s)) + 1, int(np.ceil((f["zt"] - f["z0"]) * s)) + 1


def ortho_px(v, o):
    return np.stack([(v[..., 0] - o[0][0]) / o[1], o[2] - (v[..., 1] - o[0][1]) / o[1]], -1)


def bake(b, f, s, sw):
    (w, h), img = size(f, s), sw["img"]
    hs, ws = img.shape[:2]
    F, mw = sw["floors"], sw["width"]
    z = (h - 0.5 - np.arange(h)) / s + f["z0"]
    k, fr = (z // b["fh"]).astype(int), (z % b["fh"]) / b["fh"]
    kk = np.where(k == 0, 0, 1 + (k - 1) % max(F - 1, 1)) if F > 1 else 0 * k
    ys = np.clip(hs - (kk + fr) * hs / F, 0, hs - 1).astype(int)
    m = ((np.arange(w) + 0.5) / s + f["u0"]) % (2 * mw)
    xs = np.clip(np.where(m < mw, m, 2 * mw - m) / mw * ws, 0, ws - 1).astype(int)
    return Image.fromarray(img[ys][:, xs])


def library():
    return {k: [dict(sw, img=np.asarray(Image.open(TEX / sw["file"]).convert("RGB"))) for sw in v] for k, v in json.loads((TEX / "lib.json").read_text()).items()}


def pick(lib, st, seed):
    for k in ("/".join(st), *(k for k in lib if k.startswith(f"{st[0]}/") and k.endswith(st[2])), *(k for k in lib if k.startswith(f"{st[0]}/")), "apt/old/concrete"):
        if k in lib:
            return lib[k][seed % len(lib[k])]


def model(bs, lib, ortho, A, f=1.0, pages=99, crops={}):
    jobs, (img, o) = [], ortho
    for b in bs:
        if len(b["roof"]):
            q = ortho_px(b["roof"], o)
            lo, hi = np.floor(q.reshape(-1, 2).min(0)) - 1, np.ceil(q.reshape(-1, 2).max(0)) + 1
            k = min(f * o[1] / (0.1 if f"{b['id']}:r" in crops else 0.3), (A - 16) / max(hi - lo))
            jobs.append((b, None, (lo, k), *np.ceil((hi - lo) * k).astype(int)))
        for i, w in enumerate(b["walls"]):
            c = crops.get(f"{b['id']}:{i}", ())
            s = min(f * (c[2] if len(c) > 2 else 7 if b["d"] < NEAR else 3), (A - 16) / max(w["u1"] - w["u0"], 1))
            jobs.append((b, i, s, *size(w, s)))
    if (pos := pack([j[3:] for j in jobs], A, A, pages=pages)) is None:
        return None
    out = [(Image.new("RGB", (A, A), (128, 128, 128)), [], []) for _ in range(max([p[0] for p in pos], default=0) + 1)]
    for (b, i, s, w, h), (pg, x, y) in zip(jobs, pos):
        key = f"{b['id']}:{'r' if i is None else i}"
        if i is None:
            (lo, k), t = s, b["roof"]
            im = img.crop((*lo.astype(int), *(lo + (w, h) / np.array(k)).astype(int))).resize((w, h), Image.LANCZOS)
            im = ImageEnhance.Contrast(im.filter(ImageFilter.UnsharpMask(1.5, 70, 2))).enhance(1.06)
            px = (ortho_px(t, o) - lo) * k + (x, y)
        else:
            wl, t = b["walls"][i], b["walls"][i]["tris"]
            im = bake(b, wl, s, pick(lib, b["style"], sum(map(ord, b["id"]))))
            px = np.stack([(wl["u"] if "u" in wl else t[..., :2] @ wl["e"]) * s - wl["u0"] * s + x, (wl["zt"] - t[..., 2]) * s + y], -1)
        if key in crops:
            im = Image.open(TEX / crops[key][0]).crop(tuple(crops[key][1])).resize((w, h), Image.LANCZOS)
        if i is None and "plain" in b:
            im = b["plain"](px - (x, y), w, h, k / o[1])
        out[pg][0].paste(im.resize((w + 4, h + 4)), (x - 2, y - 2))
        out[pg][0].paste(im, (x, y))
        out[pg][1].append(t.reshape(-1, 3))
        out[pg][2].append(np.c_[px.reshape(-1, 2)[:, 0] / A, px.reshape(-1, 2)[:, 1] / A])
    return [(p, np.vstack(v), np.vstack(uv)) for p, v, uv in out if v]


def glb(prims, q=80, shrink=1.0):
    lo = np.min([v.min(0) for _, v, _ in prims], 0)
    span = np.maximum(np.max([v.max(0) for _, v, _ in prims], 0) - lo, 1e-3)
    views, acc, prim, bins, images, mats = [], [], [], [], [], []

    def view(data, **kw):
        views.append({"buffer": 0, "byteOffset": sum(map(len, bins)), "byteLength": len(data), **kw})
        bins.append(data + b"\0" * (-len(data) % 4))
        return len(views) - 1

    for img, v, uv in prims:
        ext = isinstance(img, str)
        if isinstance(img, tuple):
            m = {"pbrMetallicRoughness": {"baseColorFactor": [*img[:3], 1.0], "metallicFactor": img[3], "roughnessFactor": img[4]}}
            uv = np.zeros((len(v), 0))
        else:
            if not ext:
                img.resize((round(img.width * shrink), round(img.height * shrink)), Image.LANCZOS).save(buf := io.BytesIO(), "WEBP", quality=q, method=6)
            images.append({"uri": img} if ext else {"bufferView": view(buf.getvalue()), "mimeType": "image/webp"})
            m = {"pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1}, "metallicFactor": 0, "roughnessFactor": 0.9}}
            uv = uv if ext else np.round(np.clip(uv, 0, 1) * 65535)
        mats += [m] if m not in mats else []
        w = (v - lo) / span * (1, -1, 1) + (0, 1, 0)
        a, ix = np.unique(np.c_[np.round(np.c_[w[:, [0, 2, 1]], 0 * w[:, 0]] * 65535), uv], axis=0, return_inverse=True)
        ix, t = ix.ravel().astype(np.uint16 if len(a) < 65536 else np.uint32), a[:, 4:].astype(np.float32 if ext else np.uint16)
        acc += [{"bufferView": view(ix.tobytes(), target=34963), "componentType": 5121 + ix.itemsize, "count": len(ix), "type": "SCALAR"},
                {"bufferView": view(a[:, :4].astype(np.uint16).tobytes(), byteStride=8, target=34962), "componentType": 5123, "normalized": True,
                 "count": len(a), "type": "VEC3", "min": a[:, :3].min(0).tolist(), "max": a[:, :3].max(0).tolist()}]
        attr = {"POSITION": len(acc) - 1}
        if t.shape[1]:
            acc.append({"bufferView": view(t.tobytes(), target=34962), "componentType": 5126 if ext else 5123, "normalized": not ext, "count": len(a), "type": "VEC2"})
            attr["TEXCOORD_0"] = len(acc) - 1
        prim.append({"material": mats.index(m), "indices": attr["POSITION"] - 1, "attributes": attr})
    need = ["KHR_mesh_quantization"] + ["EXT_texture_webp"] * any("bufferView" in im for im in images)
    js = {"asset": {"version": "2.0"}, "extensionsUsed": need, "extensionsRequired": need, "scene": 0,
          "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "translation": [lo[0], lo[2], -lo[1] - span[1]], "scale": [span[0], span[2], span[1]]}],
          "meshes": [{"primitives": prim}], "images": images, "materials": mats, "textures": [{"sampler": 0, **({"extensions": {"EXT_texture_webp": {"source": n}}} if "bufferView" in im else {"source": n})} for n, im in enumerate(images)],
          "samplers": [{"magFilter": 9729, "minFilter": 9987}], "accessors": acc, "bufferViews": views, "buffers": [{"byteLength": len(body := b"".join(bins))}]}
    head = (lambda h: h + b" " * (-len(h) % 4))(json.dumps(js, separators=(",", ":")).encode())
    return struct.pack("<4sIII4s", b"glTF", 2, 28 + len(head) + len(body), len(head), b"JSON") + head + struct.pack("<I4s", len(body), b"BIN\0") + body


def build():
    from data.clean import model as clean
    from data.facade import dress, plain, small
    bs, rings = buildings()
    crops = {it[0]: (sh["file"], it[1:5], *it[5:]) for f in ("near.json", "facade.json") for sh in json.loads((TEX / f).read_text())["sheets"] for it in sh["items"]}
    fine, lib, o, files = {k.split(":")[0] for k, v in crops.items() if len(v) > 2}, library(), ortho(RAW / "ortho.jpg", *DIST), []
    for b in bs:
        if b["id"] in fine:
            dress(b)
        elif b["d"] < NEAR and small(b):
            b["plain"] = plain(b)
    if outer := clean([b for b in bs if b["id"] not in fine and b["d"] >= NEAR], "../data/detail-clean/tex-"):
        (OUT / "district-outer.glb").write_bytes(outer)
        files.append("district-outer.glb")
    for name, part in (("district.glb", [b for b in bs if b["id"] not in fine and b["d"] < NEAR]), ("district-near.glb", [b for b in bs if b["id"] in fine])):
        prims = model(part, lib, o, 4096, crops=crops) if part else []
        for k in range(0, len(prims), 5):
            files.append(name if len(prims) <= 5 else name.replace(".glb", f"-{k // 5}.glb"))
            (OUT / files[-1]).write_bytes(glb(prims[k:k + 5], shrink=0.7))
    hide = [list(np.round(np.array(p.exterior.coords)[:-1], 1).ravel()) for b in bs for p in getattr(b["foot"], "geoms", [b["foot"]]) if not p.is_empty] + rings
    (OUT / "district.json").write_text(json.dumps({"models": files, "hide": hide}, separators=(",", ":")))


def tilebs(t):
    i, j = t
    wfs(lambda name: RAW / f"tiles/{name}_{i}_{j}.xml.gz", i * TS, j * TS, (i + 1) * TS, (j + 1) * TS)
    bs = [prep(b) for b in parse(read(RAW / f"tiles/lod2_{i}_{j}.xml.gz"), read(RAW / f"tiles/kanta_{i}_{j}.xml.gz"))]
    foot = [b["foot"] for b in bs]
    bs = [b for b in bs if (len(b["roof"]) or b["walls"]) and b["d"] >= FAR and np.all(b["c"] // TS == t)]
    return bs + [prep(b) for b in residual(foot, [t], lambda c: np.hypot(*(c - C)) >= FAR)[0]]


def tile(t):
    if bs := tilebs(t):
        o, lib = ortho(RAW / f"tiles/o_{t[0]}_{t[1]}.jpg", np.array(t) * TS - 100.0, 0.3, 4000), library()
        prims = next(m for A, f in [(512, 1), (1024, 1), (2048, 1)] + [(4096, 0.9 ** k) for k in range(40)] if (m := model(bs, lib, o, A, f, 1)))
        save(DET / f"{t[0]}_{t[1]}.glb", glb(prims))
    return t, len(bs)


def tiles():
    for d in (RAW / "tiles", DET):
        d.mkdir(exist_ok=True)
    zone = Polygon(np.array(json.loads((VIZ / "overview.json").read_text())["border"][0]).reshape(-1, 2)).buffer(500)
    with Pool(4) as p:
        for r in p.imap_unordered(tile, [t for t in json.loads((VIZ / "tiles.json").read_text())["tiles"] if zone.intersects(box(*np.array(t) * TS, *(np.array(t) + 1) * TS))]):
            print(*r, flush=True)
    done = sorted(tuple(map(int, f.stem.split("_"))) for f in DET.glob("*.glb"))
    part = [t for t in done if box(t[0] * TS, t[1] * TS, (t[0] + 1) * TS, (t[1] + 1) * TS).intersects(Point(C).buffer(FAR + 50))]
    (VIZ / "detail.json").write_text(json.dumps({"size": TS, "tiles": done, "partial": part}, separators=(",", ":")))


def fit(out, base):
    a, b = (Image.fromarray((np.abs(np.asarray(im, int) - BG).sum(-1) > 60).astype(np.uint8) * 255).filter(ImageFilter.MedianFilter(5)).getbbox() for im in (base, out))
    res = Image.new("RGB", out.size, BG)
    res.paste(out.crop(b).resize((a[2] - a[0], a[3] - a[1]), Image.LANCZOS), a[:2])
    return res


def call(job):
    name, img, text, refs, *raw = job
    if (TEX / f"{name}.jpg").exists():
        return Image.open(TEX / f"{name}.jpg")
    img.save(buf := io.BytesIO(), "PNG")
    key = Path.home().joinpath(".secrets/agent-secrets/openai.txt").read_text().strip()
    for _ in range(20):
        r = requests.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {key}"},
                          data={"model": "gpt-image-2.6-flare", "prompt": text, "size": "%dx%d" % img.size, "quality": "high"},
                          files=[("image[]", ("base.png", buf.getvalue(), "image/png"))] +
                          [("image[]", (p.name, p.read_bytes(), "image/png" if p.suffix == ".png" else "image/jpeg")) for p in refs], timeout=900).json()
        with open(RAW / "usage.jsonl", "a") as f:
            f.write(json.dumps({"name": name, "usage": r.get("usage"), "error": r.get("error")}) + "\n")
        if "data" in r:
            break
        time.sleep(30)
    out = Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"]))).convert("RGB").resize(img.size)
    out = out if raw else fit(out, img)
    out.save(TEX / f"{name}.jpg", quality=90)
    return out


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
