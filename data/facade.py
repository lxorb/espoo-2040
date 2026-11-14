import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from data.district import BG, ERA, NEAR, RAW, ROOT, TEX, WHAT, buildings, call, pack, prep, prism, size

PICK = ["t-1521_1136", "1036850279", "1037184105", "103718423K", "1037865518", "1014164457", "101416423H", "101416429P",
        "101426323V", "1003664713", "1014164479", "1038503111"]
PPM = 22
LOOKS = {"concrete": [((214, 212, 206), "light grey precast concrete sandwich panels with visible panel joints"),
                      ((230, 226, 216), "off-white washed exposed-aggregate concrete panels with visible panel joints"),
                      ((204, 196, 180), "light beige exposed-aggregate concrete panels with visible panel joints")],
         "brick": [((168, 82, 62), "red-brown facing brick with fine mortar courses"), ((212, 180, 120), "yellow facing brick with fine mortar courses"),
                   ((132, 76, 60), "dark brown facing brick with fine mortar courses")],
         "render": [((240, 238, 232), "smooth white render"), ((228, 214, 186), "smooth warm beige render"), ((234, 218, 160), "smooth pale yellow render"),
                    ((172, 188, 202), "smooth pale grey-blue render")],
         "wood": [((150, 70, 50), "Falu red painted vertical timber boards"), ((222, 196, 120), "pale yellow painted vertical timber boards"),
                  ((160, 160, 152), "grey painted vertical timber boards")],
         "metal": [((150, 158, 168), "grey corrugated metal sheet cladding"), ((96, 102, 110), "dark grey metal sandwich panels"),
                   ((226, 228, 226), "white metal sandwich panels")]}
CODE = {"glass": (52, 66, 86), "frame": (236, 236, 232), "door": (96, 64, 44), "sdoor": (110, 104, 96), "dock": (84, 92, 104), "hood": (70, 72, 76),
        "cap": (150, 152, 156), "gr": (150, 186, 196), "sr": (104, 110, 118)}
PROMPT = ("The first image shows {n} separate exact orthographic facade elevations of {what} in Kivenlahti, Espoo, Finland, {era}, drawn in flat colour "
          "codes. Repaint every shape as a crisp, clean, realistic architectural illustration texture of that facade, perfectly straight-on with no "
          "perspective, in the rendering style of the last reference image. Keep every outline, opening, floor line and panel joint exactly where it is; "
          "do not move, add, remove or resize anything, and keep the flat pale blue-grey background untouched. Colour codes: the main wall colour is "
          "{look}; darker band at the bottom = plinth; thin vertical dark lines = building corners and panel joints. Dark blue-grey rectangles with a light "
          "rim = windows with slim white frames, a mullion, slightly dark reflective glass and a metal sill with a thin shadow line below; long dark "
          "blue-grey areas with light vertical lines = glazed storefronts, stairwell glass strips or curtain walls with aluminium mullions; {rail}; "
          "brown = entrance doors; grey-brown = narrow storage room doors; dark grey-blue large rectangles = sectional loading doors; thin dark grey band "
          "above a door = entrance canopy; light grey band along the top = metal parapet flashing. Soft even neutral overcast daylight, no cast "
          "shadows, no sky, no people, no cars, no trees, no text, no signs, no logos.")
RAILS = {"gr": "pale teal-grey bands = frameless glass balcony railings with the dark balcony recess and balcony door above them",
         "sr": "dark grey bands = solid balcony parapet panels with the dark balcony recess and balcony door above them"}


def look(b):
    g, era, mat = b["style"]
    big = g in ("industry", "shed", "parking") or g == "shop" and b["foot"].area > 1500
    mat = "metal" if big and b["h"] < 15 else mat
    opts = LOOKS[mat]
    return opts[int(hashlib.md5(b["id"].encode()).hexdigest(), 16) % len(opts)], big, "gr" if era == "new" else "sr"


def unfold(b):
    tris = np.vstack([w["tris"] for w in b["walls"]])
    foot = max(getattr(b["foot"], "geoms", [b["foot"]]), key=lambda p: p.area)
    r = np.array(foot.simplify(0.8).exterior.coords)[:-1]
    r = r if Polygon(r).exterior.is_ccw else r[::-1]
    d = np.diff(np.vstack([r, r[:1]]), axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    turn, ln = np.abs((a - np.roll(a, 1) + np.pi) % (2 * np.pi) - np.pi), np.linalg.norm(d, axis=1)
    cut = (turn > 0.6) & (ln >= 4) & (np.roll(ln, 1) >= 4)
    k = np.flatnonzero(cut)
    r, turn, cut = (np.roll(x, -(k[0] if len(k) else 0), 0) for x in (r, turn, cut))
    ring, cum = LineString(np.vstack([r, r[:1]])), np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1))]
    cuts = np.r_[cum[np.flatnonzero(cut)] if cut.any() else [0.0], cum[-1]]
    s = lambda p: shapely.line_locate_point(ring, shapely.points(p[..., :2].reshape(-1, 2))).reshape(p.shape[:-1])
    grp = np.clip(np.searchsorted(cuts, s(tris.mean(1)), "right") - 1, 0, len(cuts) - 2)
    b["walls"] = []
    for g in np.unique(grp):
        t = tris[grp == g]
        u = s(t)
        u = np.clip(np.where(u < cuts[g] - 1, u + cum[-1], u), cuts[g], cuts[g + 1])
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, :2].sum(0)
        j = cum[:-1][(turn > 0.15) & (cum[:-1] > cuts[g] + 0.3) & (cum[:-1] < cuts[g + 1] - 0.3)]
        b["walls"].append({"tris": t, "u": u, "n": n / (np.linalg.norm(n) or 1), "joints": j})


def at(w, u, z):
    q = np.stack([w["u"][: len(w["base"])], w["base"][..., 2]], -1)
    d = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])
    dd = np.where(d == 0, 1, d)
    a = ((u - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (z - q[:, 0, 1])) / dd
    c = ((q[:, 1, 0] - q[:, 0, 0]) * (z - q[:, 0, 1]) - (u - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])) / dd
    k = np.flatnonzero((d != 0) & (a >= 0) & (c >= 0) & (a + c <= 1))
    if not len(k):
        return None
    t, k = w["base"][k[0]], k[0]
    n = np.cross(t[1] - t[0], t[2] - t[0])[:2]
    n = n / (np.linalg.norm(n) or 1)
    tg = np.array([-n[1], n[0]])
    tg = tg if (t[:, :2] - t[0, :2]) @ tg @ (w["u"][k] - w["u"][k][0]) >= 0 else -tg
    return t[0] + a[k] * (t[1] - t[0]) + c[k] * (t[2] - t[0]), n, tg


def box(w, ua, ub, za, zb, d, top=False):
    hit = at(w, (ua + ub) / 2, (za + zb) / 2)
    if hit is None:
        return [], []
    p, n, tg = hit
    q = lambda u, o, z: (np.r_[p[:2] + (u - (ua + ub) / 2) * tg + o * n, z], u)
    qs = [[q(ua, d, za), q(ub, d, za), q(ub, d, zb), q(ua, d, zb)], [q(ua, 0, za), q(ua, d, za), q(ua, d, zb), q(ua, 0, zb)],
          [q(ub, d, za), q(ub, 0, za), q(ub, 0, zb), q(ub, d, zb)]] + ([[q(ua, 0, zb), q(ua, d, zb), q(ub, d, zb), q(ub, 0, zb)]] if top else [])
    return quads(qs)


def quads(qs):
    tri = [[q[0], q[i], q[i + 1]] for q in qs for i in (1, 2)]
    return [[v for v, _ in t] for t in tri], [[u for _, u in t] for t in tri]


def parapet(w, h=0.6, d=0.25):
    qs = []
    for t, u in zip(w["base"], w["u"]):
        n = np.cross(t[1] - t[0], t[2] - t[0])
        n = np.r_[n[:2], 0] / (np.linalg.norm(n[:2]) or 1)
        for i in range(3):
            j, m = (i + 1) % 3, (i + 2) % 3
            if abs(t[i, 2] - t[j, 2]) < 0.05 and t[i, 2] > 2 and t[m, 2] < t[i, 2] - 0.3 and np.linalg.norm(t[j] - t[i]) > 0.3:
                (a, ua), (b, ub) = ((t[i], u[i]), (t[j], u[j])) if np.cross(t[j] - t[i], [0, 0, 1]) @ n > 0 else ((t[j], u[j]), (t[i], u[i]))
                up, ins = np.array([0, 0, h]), -d * n
                qs += [[(a, ua), (b, ub), (b + up, ub), (a + up, ua)], [(b + ins, ub), (a + ins, ua), (a + ins + up, ua), (b + ins + up, ub)],
                       [(a + up, ua), (b + up, ub), (b + ins + up, ub), (a + ins + up, ua)]]
    return quads(qs)


def plan(b, w, role, big):
    g, fh, u0, u1 = b["style"][0], b["fh"], w["u0"], w["u1"]
    width, nf, res = u1 - u0, max(1, b["floors"] or round(b["h"] / 3)), g in ("apt", "row", "house")
    top, out = min(w["zt"], nf * fh), []
    if big:
        out.append(("glass", u0 + 1, u1 - 1, top - 1.6, top - 0.6)) if width > 8 else None
        for k in range(int(width // 14) if role == "entry" else 0):
            u = u0 + (k + 0.5) * width / int(width // 14)
            out += [("dock", u - 1.8, u + 1.8, 0.0, 3.8), ("hood", u - 2.3, u + 2.3, 4.0, 4.3)]
        if width > 6:
            out += [("door", u0 + 2, u0 + 3, 0.2, 2.3), ("hood", u0 + 1.7, u0 + 3.3, 2.5, 2.75)]
        return out
    if g in ("shop", "office", "public", "transit") and width > 4:
        out += [("glass", u0 + 0.4, u1 - 0.4, 0.4, min(fh, top) - 0.4)] + [("door", u - 0.9, u + 0.9, 0.2, 2.5) for u in np.linspace(u0, u1, 4)[1:-1] if role == "entry"]
    bays = max(1, round(width / (3.4 if res else 3.0)))
    bw = width / bays
    nd = max(1, round(width / (8 if g in ("row", "house") else 18)))
    door = set(np.round((np.arange(nd) + 0.5) * bays / nd - 0.5).astype(int)) if role == "entry" and width > 6 else set()
    cols = range(bays) if role != "gable" else [bays // 2] if width >= 8 or g in ("row", "house") else []
    for i in cols:
        u = u0 + (i + 0.5) * bw
        if i in door and g == "apt" and top > 6:
            out.append(("glass", u - 0.7, u + 0.7, 3.2, top - 0.8))
        for f in range(1 if g in ("shop", "office", "public", "transit") else 0, nf):
            z = f * fh
            if z + fh > top + 0.1:
                break
            if f == 0 and i in door:
                out += [("door", u - 0.8, u + 0.8, 0.2, 2.4), ("hood", u - 1.3, u + 1.3, 2.6, 2.85)]
            elif i in door and g == "apt":
                continue
            elif f == 0 and role == "entry" and g == "apt" and i % 3 == 2:
                out.append(("sdoor", u - 0.5, u + 0.5, 0.2, 2.2))
            elif role == "bal" and (f > 0 or g == "row") and i % (3 if g == "apt" else 2) == 1:
                out += [("dark", u - 1.7, u + 1.7, z + 0.1, z + fh - 0.3), ("rail", u - 1.7, u + 1.7, z + 0.1, z + 1.2)]
            elif width >= 4:
                out.append(("win", u - 0.8, u + 0.8, z + 0.8, z + 2.3) if g != "office" else ("glass", u - bw / 2 + 0.2, u + bw / 2 - 0.2, z + 0.8, z + 2.6))
    return out


def flat(b):
    r = b["roof"]
    n = np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]) if len(r) else np.zeros((0, 3))
    a = np.linalg.norm(n, axis=1)
    return len(r) > 0 and a[n[:, 2] > 0.97 * a].sum() > 0.8 * a.sum()


def small(b):
    return b["foot"].area < 200 and flat(b)


def plain(b):
    seed = int(hashlib.md5(b["id"].encode()).hexdigest(), 16) % 2 ** 32

    def draw(t, w, h, ppm):
        rng = np.random.default_rng(seed)
        img = Image.fromarray(np.clip(np.full((h, w, 3), (70, 72, 76)) + rng.normal(0, 3, (h, w, 1)), 0, 255).astype(np.uint8))
        d, poly = ImageDraw.Draw(img), unary_union([Polygon(q).buffer(0.5) for q in t]).buffer(-0.5)
        for y in np.arange(rng.uniform(0, 1.5 * ppm), h, 1.5 * ppm):
            d.line([(0, y), (w, y)], fill=(62, 64, 68), width=1)
        for p in getattr(poly, "geoms", [poly]):
            d.line(list(p.exterior.coords), fill=(176, 178, 180), width=max(2, round(0.35 * ppm)))
            for c in [p.representative_point()] + ([p.centroid] if p.area > 60 * ppm ** 2 else []):
                v = 0.45 * ppm
                d.rectangle([c.x - v, c.y - v, c.x + v, c.y + v], fill=(152, 154, 156), outline=(60, 60, 62))
        return img
    return draw


def boxify(b):
    p = max(getattr(b["foot"], "geoms", [b["foot"]]), key=lambda p: p.area).simplify(0.3)
    b["faces"] = prism(p.exterior.coords, float(b["roof"][..., 2].max()))
    prep(b)
    b["plain"] = plain(b)


def dress(b):
    if small(b):
        boxify(b)
    unfold(b)
    (col, desc), big, rail = look(b)
    ln = [w["u"].max() - w["u"].min() for w in b["walls"]]
    sw = [w["n"] @ [-0.6, -0.8] for w in b["walls"]]
    long = [k for k in range(len(ln)) if ln[k] >= 0.6 * max(ln)]
    bal = max(long, key=lambda k: sw[k]) if b["style"][0] in ("apt", "row") and max(ln) >= 12 else -1
    entry = min(long, key=lambda k: sw[k]) if len(long) > 1 or bal < 0 else -1
    b["look"] = col, desc, big, rail
    for k, w in enumerate(b["walls"]):
        w["base"], w["u0"], w["u1"], w["zt"] = w["tris"], w["u"].min(), w["u"].max(), w["tris"][..., 2].max()
        w["plan"] = plan(b, w, "bal" if k == bal else "entry" if k == entry else "side" if k in long or ln[k] >= 14 else "gable", big)
        add = [box(w, *c, 1.0 if kind == "rail" else 1.4, kind == "hood") for kind, *c in w["plan"] if kind in ("rail", "hood")]
        cap = parapet(w, 0.3 if small(b) else 0.6) if flat(b) else ([], [])
        w["cap"] = np.array(cap[0]).reshape(-1, 3, 3)
        w["tris"] = np.vstack([w["base"], *(np.array(x[0]).reshape(-1, 3, 3) for x in add + [cap] if x[0])])
        w["u"] = np.vstack([w["u"], *(np.array(x[1]).reshape(-1, 3) for x in add + [cap] if x[1])])
        w["u0"], w["u1"], w["z0"] = w["u"].min(), w["u"].max(), w["tris"][..., 2].min()
        w["zt"] = max(w["tris"][..., 2].max(), w["z0"] + 0.5)


def control(b, w, s):
    (W, H), (col, _, _, rail) = size(w, s), b["look"]
    px = lambda u, z: ((u - w["u0"]) * s, (w["zt"] - z) * s)
    m, layer, img = Image.new("L", (W, H), 0), Image.new("RGB", (W, H), col), Image.new("RGB", (W, H), BG)
    for t, u in zip(w["base"], w["u"]):
        ImageDraw.Draw(m).polygon([px(uu, p[2]) for p, uu in zip(t, u)], fill=255)
    d, dark, lw = ImageDraw.Draw(layer), tuple(int(c * 0.62) for c in col), max(1, round(s * 0.08))
    d.rectangle([0, min(px(0, 0.6)[1], H), W, H], fill=tuple(int(c * 0.72) for c in col))
    for z in np.arange(b["fh"], w["zt"], b["fh"]):
        d.line([(0, px(0, z)[1]), (W, px(0, z)[1])], fill=tuple(int(c * 0.84) for c in col), width=lw)
    for u in w["joints"]:
        d.line([(px(u, 0)[0], 0), (px(u, 0)[0], H)], fill=dark, width=lw + 1)
    inside = lambda c: c[1] > c[0] and c[3] > c[2] and all(0 <= x < W and 0 <= y < H and m.getpixel((int(x), int(y))) for x, y in (px(c[0], c[2]), px(c[1], c[2]), px(c[0], c[3] - 0.01), px(c[1], c[3] - 0.01)))
    for kind, *c in w["plan"]:
        if inside(c):
            (x0, y1), (x1, y0) = px(c[0], c[2]), px(c[1], c[3])
            if kind in ("win", "glass"):
                d.rectangle([x0, y0, x1, y1], fill=CODE["frame"])
                d.rectangle([x0 + s * 0.08, y0 + s * 0.08, x1 - s * 0.08, y1 - s * 0.08], fill=CODE["glass"])
                for x in np.arange(x0 + s * 1.5, x1 - s * 0.5, s * 1.5) if kind == "glass" else [(x0 + x1) / 2]:
                    d.line([(x, y0), (x, y1)], fill=CODE["frame"], width=lw)
                if kind == "win":
                    d.line([(x0 - s * 0.1, y1 + lw), (x1 + s * 0.1, y1 + lw)], fill=dark, width=lw)
            else:
                d.rectangle([x0, y0, x1, y1], fill=CODE[rail if kind == "rail" else "glass" if kind == "dark" else kind])
    img.paste(layer, mask=m)
    img.paste(Image.new("RGB", (W, H), (70, 70, 70)), mask=m.filter(ImageFilter.FIND_EDGES))
    k = len(w["tris"]) - len(w["cap"])
    for t, u in zip(w["tris"][k:], w["u"][k:]):
        ImageDraw.Draw(img).polygon([px(uu, p[2]) for p, uu in zip(t, u)], fill=CODE["cap"])
    return img


def scale(w, s=PPM):
    return min(s, 1500 / max(w["u1"] - w["u0"], 1), 990 / max(w["zt"] - w["z0"], 1))


def layout(ws):
    fits = [(max(p[0] for p in pos) + 1, -s, s, pos) for s in range(PPM, 15, -1) if (pos := pack([size(w, scale(w, s)) for w in ws], 1536, 1024, 8))]
    return min(fits)[2:]


def ranked(bs):
    c = [b for b in bs if b["d"] < NEAR and not b["foot"].is_empty and b["walls"] and b["foot"].area > 100]
    area = lambda b: sum((w["u1"] - w["u0"]) * (w["zt"] - w["z0"]) for w in b["walls"])
    return sorted(c, key=lambda b: (b["id"] not in PICK, -area(b) / (b["d"] + 150)))


def sheets(n, *only):
    jobs, book = [], {"sheets": []}
    for b in ranked(buildings()[0])[: int(n)]:
        dress(b)
        ws = list(enumerate(b["walls"]))
        s, pos = layout(b["walls"])
        for pg in range(max(p[0] for p in pos) + 1):
            base, its = Image.new("RGB", (1536, 1024), BG), []
            for (i, w), (p, x, y) in zip(ws, pos):
                if p == pg:
                    base.paste(control(b, w, scale(w, s)), (x, y))
                    its.append([f"{b['id']}:{i}", x, y, x + size(w, scale(w, s))[0], y + size(w, scale(w, s))[1], scale(w, s)])
            name = "fac_" + hashlib.md5(json.dumps(its).encode() + bytes(np.asarray(base))).hexdigest()[:8]
            book["sheets"].append({"file": f"{name}.jpg", "items": its})
            text = PROMPT.format(n=len(its), what=WHAT[b["style"][0]], era=ERA[b["style"][1]], look=b["look"][1], rail=RAILS[b["look"][3]])
            jobs.append((name, base, text, sorted((RAW / "photos").glob(f"{b['style'][0]}_{b['style'][1]}*.jpg")) + [ROOT / "data/hype/hype_s.jpg"]))
    for j in jobs:
        j[1].save(RAW / f"{j[0]}_in.png")
    jobs = [j for j in jobs if not only or j[0].startswith(only)]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(call, jobs))
    (TEX / "facade.json").write_text(json.dumps(book))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
