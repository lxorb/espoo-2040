import base64
import hashlib
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from pyproj import Transformer
from shapely import constrained_delaunay_triangles
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from data.geo import ROOT, VIZ
from data.hype import HYPE_NW, HYPE_SE, HYPE_SW, PARKING

RAW, TEX, OUT = ROOT / "data/raw/district", ROOT / "data/district", ROOT / "viz/assets"
WFS = "https://kartat.espoo.fi/teklaogcweb/wfs.ashx"
WMS = "https://kartat.espoo.fi/teklaogcweb/wms.ashx"
C = np.array([-1511.7, 1239.2])
NEAR, FAR = 600, 1500
OWN = unary_union([Polygon([HYPE_SW, HYPE_SE, (HYPE_SE[0] + HYPE_NW[0] - HYPE_SW[0], HYPE_SE[1] + HYPE_NW[1] - HYPE_SW[1]), HYPE_NW]), Polygon(PARKING)]).buffer(3)
O0, OPX, ON = C - 1560, 0.3, 10400
GK = Transformer.from_crs(3879, 3067, always_xy=True)
NS = {"gml": "http://www.opengis.net/gml", "bldg": "http://www.opengis.net/citygml/building/2.0",
      "gen": "http://www.opengis.net/citygml/generics/2.0"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    x0, y0 = Transformer.from_crs(3067, 3879, always_xy=True).transform(*(O0 + (370000, 6670000)))
    box = f"{x0 - 50:.0f},{y0 - 50:.0f},{x0 + 3250:.0f},{y0 + 3250:.0f}"
    for name, typ, ver in (("lod2", "bldg:building_lod2", "1.1.0"), ("kanta", "kanta:Rakennus", "1.0.0")):
        if not (RAW / f"{name}.xml").exists():
            r = requests.get(WFS, params={"SERVICE": "WFS", "VERSION": ver, "REQUEST": "GetFeature", "TYPENAME": typ,
                                          "BBOX": box, "SRSNAME": "EPSG:3879"}, timeout=900)
            (RAW / f"{name}.xml").write_bytes(r.content)
    if (RAW / "ortho.jpg").exists():
        return
    img, step = Image.new("RGB", (ON, ON)), 2080

    def tile(ij):
        i, j = ij
        e, n = O0 + (370000, 6670000) + (i * step * OPX, (ON - (j + 1) * step) * OPX)
        r = requests.get(WMS, params={"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": "Ortokartta",
                                      "STYLES": "", "SRS": "EPSG:3067", "FORMAT": "image/jpeg", "WIDTH": step, "HEIGHT": step,
                                      "BBOX": f"{e},{n},{e + step * OPX},{n + step * OPX}"}, timeout=300)
        img.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), (i * step, j * step))

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(tile, [(i, j) for i in range(ON // step) for j in range(ON // step)]))
    img.save(RAW / "ortho.jpg", quality=92)


def parse():
    mats = dict(re.findall(r"<kanta:rakennustunnus>(\w+)</kanta:rakennustunnus>.*?<kanta:julkisivumateriaali>([^<]*)<",
                           (RAW / "kanta.xml").read_text(), flags=re.S))
    out = []
    for _, el in ET.iterparse(RAW / "lod2.xml"):
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
                    rings.append(np.c_[x - 370000, y - 6670000, v[:, 2]])
                faces.append((s.tag.split("}")[1][0], rings))
        a["h"] = el.findtext("bldg:measuredHeight", "0", NS)
        el.clear()
        i = a.get("rakennustunnus") or "x%.0f_%.0f" % tuple(faces[0][1][0][0, :2]) if faces else ""
        if faces:
            out.append({"id": i, "use": a.get("kayttotarkoitus") or "", "year": int((a.get("valmistumispvm") or "0")[:4]),
                        "floors": int(a.get("kerrosluku") or 0), "mat": mats.get(i, ""), "faces": faces,
                        "h": float(a["h"])})
    return out


GROUPS = [("house", "Omakoti|Pari|Sauna|vapaa-ajan"), ("row", "Rivi"), ("apt", "kerrosta|Erityis|Asuntola"),
          ("shop", "Kauppa|myymälä|Ravintola|majoitus"), ("office", "Toimisto"), ("parking", "Pysäköinti"),
          ("public", "päiväkod|oppilai|Urheilu|urheilu|kokoontumis|Seura|Uskonnon|Laitos|hoito"),
          ("industry", "teollisuus|Lämpimät"), ("transit", "liikenteen")]
BKIND = {"residential": "house", "apartments": "apt", "commercial": "shop", "industrial": "industry", "public": "public",
         "garage": "parking", "other": "shed"}
MAT = {"puu": "wood", "tiili": "brick", "betoni": "concrete", "metallilevy": "metal"}
TINT = {"wood": (196, 170, 128), "brick": (160, 96, 78), "concrete": (200, 198, 192), "metal": (150, 156, 162),
        "render": (222, 214, 196)}
BG, A = (205, 214, 224), 4096


def style(b):
    g = b["use"] if b["use"] in BKIND.values() else next((g for g, p in GROUPS if re.search(p, b["use"])),
                                                         "apt" if b["floors"] >= 3 else "industry" if not b["use"] and b["h"] > 7 else "shed")
    era = "old" if b["year"] < 1990 else "mid" if b["year"] < 2010 else "new"
    return g, era, MAT.get(b["mat"], "render")


def triangulate(rings):
    r0 = rings[0]
    n = sum(np.cross(r0[i], r0[(i + 1) % len(r0)]) for i in range(len(r0)))
    if np.linalg.norm(n) < 1e-6:
        return np.zeros((0, 3, 3)), np.zeros(3)
    n = n / np.linalg.norm(n)
    a = np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    a /= np.linalg.norm(a)
    b, o = np.cross(n, a), r0[0]
    rings = [np.c_[(r - o) @ a, (r - o) @ b] for r in rings if len(np.unique(r.round(3), axis=0)) > 2]
    p, out = Polygon(rings[0], rings[1:]) if rings and len(rings[0]) > 2 else Polygon(), []
    for t in constrained_delaunay_triangles(p if p.is_valid else p.buffer(0)).geoms:
        q = np.array(t.exterior.coords)[:3]
        v = o + q[:, :1] * a + q[:, 1:] * b
        out.append(v if np.cross(v[1] - v[0], v[2] - v[0]) @ n > 0 else v[::-1])
    return np.array(out).reshape(-1, 3, 3), n


def prism(ring, h):
    ring = np.asarray(ring)[:-1] if np.allclose(ring[0], ring[-1]) else np.asarray(ring)
    if not Polygon(ring).exterior.is_ccw:
        ring = ring[::-1]
    walls = [("W", [np.array([(*a, 0), (*b, 0), (*b, h), (*a, h)])]) for a, b in zip(ring, np.roll(ring, -1, 0))]
    return walls + [("R", [np.c_[ring, np.full(len(ring), h)]])]


def residual(foot):
    taken, out, rings, kinds = unary_union(foot), [], [], json.loads((VIZ / "tiles.json").read_text())["bkinds"]
    for f in VIZ.glob("tiles/*.json"):
        i, j = map(int, f.stem.split("_"))
        if np.hypot((i + 0.5) * 1000 - C[0], (j + 0.5) * 1000 - C[1]) > FAR + 800:
            continue
        for h, k, xy in json.loads(f.read_text())["b"]:
            ring, p = np.array(xy).reshape(-1, 2), Polygon(np.array(xy).reshape(-1, 2)).buffer(0)
            if np.hypot(*(ring.mean(0) - C)) >= FAR:
                continue
            inner = p.contains(Point(ring.mean(0)))
            rings.append(list(np.round(ring if inner else ring.mean(0) + [(-1, -1), (1, -1), (1, 1), (-1, 1)], 1).ravel()))
            if p.area > 4 and p.intersection(taken).area < 0.3 * p.area:
                out.append({"id": f"t{ring[0][0]:.0f}_{ring[0][1]:.0f}", "use": BKIND[kinds[k]], "year": 2020,
                            "floors": round(h / 3), "mat": "", "faces": prism(ring, h), "foot": p, "h": h})
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
    eu = max((mrr[1] - mrr[0], mrr[2] - mrr[1]), key=np.linalg.norm)
    eu = eu / (np.linalg.norm(eu) or 1)
    axes = [eu, np.array([-eu[1], eu[0]]), -eu, np.array([eu[1], -eu[0]])]
    tops = [(Polygon(r[0][:, :2]).buffer(0), r[0][:, 2].max()) for k, r in faces if k == "R" and len(r[0]) > 2]
    sides, roof = [{"e": axes[(k + 1) % 4], "tris": [], "rings": []} for k in range(4)], []
    for k, rs in faces:
        rs = [np.c_[r[:, :2], np.maximum(r[:, 2], 0)] for r in rs]
        t, n = triangulate(rs)
        if k != "W" and abs(n[2]) > 0.3:
            roof.append(t if n[2] > 0 else t[:, ::-1])
        elif len(t):
            c = rs[0][:, :2].mean(0)
            q = Point(c + 0.4 * n[:2] / (np.linalg.norm(n[:2]) or 1))
            if any(z > rs[0][:, 2].mean() and p.contains(q) for p, z in tops):
                t, n = t[:, ::-1], -n
            w = sides[int(np.argmax([n[:2] @ x for x in axes]))]
            w["tris"].append(t)
            w["rings"].append(rs[0])
    b["walls"] = [w for w in sides if w["tris"]]
    for w in b["walls"]:
        w["tris"] = np.vstack(w["tris"])
        u = w["tris"][..., :2] @ w["e"]
        w["u0"], w["u1"], w["z0"] = u.min(), u.max(), w["tris"][..., 2].min()
        w["zt"] = max(w["tris"][..., 2].max(), w["z0"] + 0.5)
    b["roof"] = np.vstack(roof) if roof else np.zeros((0, 3, 3))
    b["d"] = float(np.hypot(*(np.array(b["foot"].centroid.coords[0] if not b["foot"].is_empty else pts[:, :2].mean(0)) - C)))
    b["style"] = style(b)
    b["fh"] = float(np.clip(b["h"] / b["floors"], 2.7, 4.5)) if b["floors"] > 0 and b["h"] > 0 else 3.0
    return b


def buildings():
    bs = [prep(b) for b in parse() if np.hypot(*(b["faces"][0][1][0][:, :2].mean(0) - C)) < FAR + 100]
    bs = [b for b in bs if b["d"] < FAR and (len(b["roof"]) or b["walls"])]
    extra, rings = residual([b["foot"] for b in bs])
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


def draw(b, f, s, bg=None):
    (w, h), col = size(f, s), TINT[b["style"][2]]
    im, m = Image.new("RGB", (w, h), col), Image.new("L", (w, h), 0)
    for z in np.arange(b["fh"], f["zt"], b["fh"]):
        ImageDraw.Draw(im).line([(0, (f["zt"] - z) * s), (w, (f["zt"] - z) * s)], fill=tuple(int(c * 0.75) for c in col))
    for r in f["rings"]:
        ImageDraw.Draw(m).polygon([((p[:2] @ f["e"] - f["u0"]) * s, (f["zt"] - p[2]) * s) for p in r], fill=255, outline=255)
    im = Image.composite(Image.new("RGB", (w, h), (70, 70, 70)), im, m.filter(ImageFilter.FIND_EDGES))
    return Image.composite(im, Image.new("RGB", (w, h), bg or col), m if bg else Image.new("L", (w, h), 255))


def ortho_px(v):
    return np.stack([(v[..., 0] - O0[0]) / OPX, ON - (v[..., 1] - O0[1]) / OPX], -1)


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
    lib = json.loads((TEX / "lib.json").read_text()) if (TEX / "lib.json").exists() else {}
    for v in lib.values():
        v[:] = [dict(sw, img=np.asarray(Image.open(TEX / sw["file"]).convert("RGB"))) for sw in v if (TEX / sw["file"]).exists()]
    return {k: v for k, v in lib.items() if v}


def pick(lib, st, seed):
    for k in ("/".join(st), *(k for k in lib if k.startswith(f"{st[0]}/") and k.endswith(st[2])),
              *(k for k in lib if k.startswith(f"{st[0]}/")), "apt/old/concrete"):
        if k in lib:
            return lib[k][seed % len(lib[k])]


def build():
    Image.MAX_IMAGE_PIXELS = None
    bs, rings = buildings()
    lib, ortho = library(), Image.open(RAW / "ortho.jpg")
    near = json.loads((TEX / "near.json").read_text()) if (TEX / "near.json").exists() else {"sheets": []}
    crops = {it[0]: (sh["file"], it[1:]) for sh in near["sheets"] for it in sh["items"] if (TEX / sh["file"]).exists()}
    jobs = []
    for b in bs:
        if len(b["roof"]):
            q = ortho_px(b["roof"])
            o, k = np.floor(q.reshape(-1, 2).min(0)) - 1, 1 if b["d"] < NEAR else 0.5
            jobs.append((b, None, (o, k), *np.ceil((np.ceil(q.reshape(-1, 2).max(0)) + 1 - o) * k).astype(int)))
        for i, f in enumerate(b["walls"]):
            s = min(7 if b["d"] < NEAR else 3, (A - 16) / max(f["u1"] - f["u0"], 1))
            jobs.append((b, i, s, *size(f, s)))
    pos = pack([j[3:] for j in jobs], A, A)
    pages = [Image.new("RGB", (A, A), (128, 128, 128)) for _ in range(max(p[0] for p in pos) + 1)]
    geo = [([], []) for _ in pages]
    for (b, i, s, w, h), (pg, x, y) in zip(jobs, pos):
        if i is None:
            (o, k), t = s, b["roof"]
            img = ortho.crop((*o.astype(int), *(o + (w, h) / np.array(k)).astype(int))).resize((w, h), Image.LANCZOS)
            px = (ortho_px(t) - o) * k + (x, y)
        else:
            f, key = b["walls"][i], f"{b['id']}:{i}"
            if key in crops:
                img = Image.open(TEX / crops[key][0]).crop(tuple(crops[key][1])).resize((w, h), Image.LANCZOS)
            else:
                sw = pick(lib, b["style"], sum(map(ord, b["id"])))
                img = bake(b, f, s, sw) if sw else draw(b, f, s)
            t = f["tris"]
            px = np.stack([(t[..., :2] @ f["e"] - f["u0"]) * s + x, (f["zt"] - t[..., 2]) * s + y], -1)
        pages[pg].paste(img.resize((w + 4, h + 4)), (x - 2, y - 2))
        pages[pg].paste(img, (x, y))
        geo[pg][0].append(t.reshape(-1, 3))
        geo[pg][1].append(np.c_[px.reshape(-1, 2)[:, 0] / A, 1 - px.reshape(-1, 2)[:, 1] / A])
    meshes = []
    for n, (page, (v, uv)) in enumerate(zip(pages, geo)):
        buf = io.BytesIO()
        page.save(buf, "JPEG", quality=84)
        v = np.vstack(v)
        mat = trimesh.visual.material.PBRMaterial(name=f"district{n}", baseColorTexture=Image.open(buf),
                                                  metallicFactor=0.0, roughnessFactor=0.9)
        meshes.append(trimesh.Trimesh(np.c_[v[:, 0], v[:, 2], -v[:, 1]], np.arange(len(v)).reshape(-1, 3), process=False,
                                      visual=trimesh.visual.TextureVisuals(uv=np.vstack(uv), material=mat)))
        meshes[-1].merge_vertices(merge_tex=False, merge_norm=True)
    for old in OUT.glob("district*.glb"):
        old.unlink()
    groups, used = [[]], 0
    for m in meshes:
        n = len(trimesh.Scene(m).export(file_type="glb"))
        if groups[-1] and used + n > 24e6:
            groups, used = groups + [[]], 0
        groups[-1].append(m)
        used += n
    files = ["district.glb"] if len(groups) == 1 else [f"district-{k}.glb" for k in range(len(groups))]
    for name, g in zip(files, groups):
        (OUT / name).write_bytes(trimesh.Scene(g).export(file_type="glb"))
    hide = [list(np.round(np.array(p.exterior.coords)[:-1], 1).ravel()) for b in bs
            for p in getattr(b["foot"], "geoms", [b["foot"]]) if not p.is_empty] + rings
    (OUT / "district.json").write_text(json.dumps({"models": files, "hide": hide}, separators=(",", ":")))
    print(files, len(bs), sum(b["d"] < NEAR for b in bs), sum(len(m.faces) for m in meshes), "tris", len(pages), "pages")


WHAT = {"house": "detached family houses", "row": "terraced row houses", "apt": "apartment blocks",
        "shed": "small outbuildings, storage sheds and garages", "shop": "retail buildings", "office": "office buildings",
        "public": "public buildings such as a school, day care centre or sports hall", "parking": "parking garages",
        "industry": "industrial halls and warehouses", "transit": "metro station entrance and ventilation buildings"}
ERA = {"old": "built in the 1970s and 1980s", "mid": "built in the 1990s and 2000s", "new": "built after 2010"}
SKIN = {"wood": "painted timber board cladding", "brick": "brick", "concrete": "white or light grey precast concrete panels",
        "metal": "profiled metal sheet cladding", "render": "rendered or mixed panel cladding"}
RULES = ("Keep every shape's outline, size and position exactly; do not add, merge, move or resize shapes and keep the "
         "flat pale blue-grey background untouched. Perfectly straight-on with no perspective, soft even overcast daylight, "
         "no cast shadows, no sky, no people, no cars, no trees or plants in front, no text, no signs, no logos.")


def prompt(st, what):
    return (f"The first image shows {what} of {WHAT[st[0]]} in Kivenlahti, Espoo, Finland, {ERA[st[1]]}, with {SKIN[st[2]]} "
            "facades. Each flat-coloured shape is one wall seen exactly straight-on with its bottom edge at ground level; "
            "thin dark lines mark the floor levels. Repaint every shape as a realistic photographic facade texture typical "
            "of such Finnish buildings: windows aligned to the floor lines, entrances only along the bottom edge, "
            "balconies, drainpipes and a plinth where typical. " + RULES + (" The other images are real photographs of such "
            "buildings in Kivenlahti for material and colour reference only." if st[0] in ("apt", "transit") else ""))


def call(job):
    name, img, text, st = job
    refs = sorted((RAW / "photos").glob(f"{st[0]}_{st[1]}*.jpg"))
    if (TEX / f"{name}.jpg").exists():
        return Image.open(TEX / f"{name}.jpg")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    key = Path.home().joinpath(".secrets/agent-secrets/openai.txt").read_text().strip()
    for _ in range(20):
        r = requests.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {key}"},
                          data={"model": "gpt-image-2.6-flare", "prompt": text, "size": "1536x1024", "quality": "high"},
                          files=[("image[]", ("base.png", buf.getvalue(), "image/png"))] +
                          [("image[]", (p.name, p.read_bytes(), "image/jpeg")) for p in refs], timeout=900).json()
        with open(RAW / "usage.jsonl", "a") as f:
            f.write(json.dumps({"name": name, "usage": r.get("usage"), "error": r.get("error")}) + "\n")
        if "data" in r:
            break
        time.sleep(30)
    out = Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"]))).convert("RGB").resize((1536, 1024))
    out.save(TEX / f"{name}.jpg", quality=90)
    return out


def sheets(*only):
    TEX.mkdir(exist_ok=True)
    bs, _ = buildings()
    own = lambda b: b["d"] < NEAR and b["id"][0] not in "tx" and (b["style"][0] != "shed" or b["foot"].area > 150)
    items = sorted(((b, i, f) for b in bs if own(b) for i, f in enumerate(b["walls"])),
                   key=lambda t: (t[0]["style"][::2], round(np.arctan2(*(np.array(t[0]["foot"].centroid.coords[0]) - C)), 1),
                                  t[0]["id"], t[1]))
    groups, cur = [], []
    for it in items:
        if cur and (cur[0][0]["style"][::2] != it[0]["style"][::2] or not pack([size(f, 12) for *_, f in cur + [it]], 1536, 1024, 8, 1)):
            groups.append(cur)
            cur = []
        cur.append(it)
    jobs, layout, lib = [], {"sheets": []}, {}
    for n, g in enumerate(groups + [cur]):
        s = next(s for s in range(16, 0, -1) if pack([size(f, s) for *_, f in g], 1536, 1024, 8, 1))
        pos, base = pack([size(f, s) for *_, f in g], 1536, 1024, 8, 1), Image.new("RGB", (1536, 1024), BG)
        for (b, i, f), (_, x, y) in zip(g, pos):
            base.paste(draw(b, f, s, BG), (x, y))
        items = [[f"{b['id']}:{i}", x, y, x + size(f, s)[0], y + size(f, s)[1]] for (b, i, f), (_, x, y) in zip(g, pos)]
        name = "near_" + hashlib.md5(json.dumps(items).encode()).hexdigest()[:8]
        layout["sheets"].append({"file": f"{name}.jpg", "items": items})
        eras = [b["style"][1] for b, *_ in g]
        st = (g[0][0]["style"][0], max(eras, key=eras.count), g[0][0]["style"][2])
        jobs.append((name, base, prompt(st, f"{len(g)} separate flat orthographic facade elevations"), st))
    area = {}
    for b in bs:
        if b["d"] >= NEAR:
            area[b["style"]] = area.get(b["style"], 0) + sum((f["u1"] - f["u0"]) * (f["zt"] - f["z0"]) for f in b["walls"])
    top = sorted(area, key=area.get, reverse=True)
    for st in top[:18] + [st for st in top[18:] if st == next(t for t in top if t[0] == st[0])]:
        F = {"apt": 4, "office": 4, "parking": 3, "shed": 1}.get(st[0], 2)
        base, name = Image.new("RGB", (1536, 1024), BG), "lib_" + "_".join(st)
        fake = {"style": st, "fh": 3.0}
        for y in (30, 534):
            sw = {"u0": 0, "u1": 1500 / (460 / (F * 3)), "z0": 0, "zt": F * 3.0}
            sw["rings"] = [np.array([(0, 0, 0), (sw["u1"], 0, 0), (sw["u1"], 0, sw["zt"]), (0, 0, sw["zt"])])]
            sw["e"] = np.array([1.0, 0])
            base.paste(draw(fake, sw, 460 / (F * 3), BG).resize((1500, 460)), (18, y))
        lib["/".join(st)] = [{"file": f"{name}_{k}.jpg", "floors": F, "width": 1500 / (460 / (F * 3)), "box": [18, y, 1518, y + 460]}
                             for k, y in enumerate((30, 534))]
        jobs.append((name, base, prompt(st, "two separate flat orthographic facade swatches, each exactly "
                                            f"{F} storeys and the upper and lower swatch from two different buildings, "
                                            "with regular repeating window bays and no building corners"), st))
    for j in jobs:
        j[1].save(RAW / f"{j[0]}_in.png")
    jobs = [j for j in jobs if not only or j[0] in only]
    with ThreadPoolExecutor(4) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(call, jobs)))
    for k, v in lib.items():
        for sw in v:
            if sw["file"].rsplit("_", 1)[0] in res:
                res[sw["file"].rsplit("_", 1)[0]].crop(tuple(sw["box"])).save(TEX / sw["file"], quality=90)
    (TEX / "near.json").write_text(json.dumps(layout))
    (TEX / "lib.json").write_text(json.dumps(lib))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
