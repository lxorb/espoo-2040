import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from data.district import BG, ERA, RAW, ROOT, TEX, WHAT, buildings, call, pack, size

PICK = ["t-1521_1136", "1036850279", "1037184105", "103718423K", "1037865518", "1014164457", "101416423H", "101416429P",
        "101426323V", "1003664713", "1014164479", "1038503111"]
PPM = 22
SKIN = {"concrete": (232, 230, 222), "brick": (168, 82, 62), "render": (226, 212, 186), "wood": (186, 140, 96), "metal": (150, 158, 168)}
GLASS, FRAME, RAIL, DOOR, HOOD, CAP = (52, 66, 86), (236, 236, 232), (120, 128, 136), (96, 64, 44), (70, 72, 76), (150, 152, 156)
LOOK = {"concrete": "white or light grey precast concrete sandwich panels with regular vertical and horizontal panel joints",
        "brick": "red-brown facing brick with fine mortar courses", "render": "smooth light rendered plaster", "wood": "painted vertical timber boards",
        "metal": "profiled metal sheet cladding"}
PROMPT = ("The first image shows {n} separate exact orthographic facade elevations of {what} in Kivenlahti, Espoo, Finland, {era}, drawn in flat colour "
          "codes. Repaint every shape as a crisp, clean, realistic architectural illustration texture of that facade, perfectly straight-on with no "
          "perspective, in the rendering style of the last reference image. Keep every outline, opening, floor line and panel boundary exactly where it is; "
          "do not move, add, remove or resize anything, and keep the flat pale blue-grey background untouched. Colour codes: the main wall colour = {look}; "
          "dark blue-grey rectangles with a light rim = windows with slim white or aluminium frames, mullions and slightly dark reflective glass; "
          "large dark blue-grey areas = glazed storefronts or curtain walls with aluminium mullions; mid grey bands = balcony parapets of glass or steel "
          "with the dark balcony recess and balcony door above them; brown = entrance doors; thin dark grey band above a door = entrance canopy; "
          "light grey band along the top = metal parapet flashing; darker band at the bottom = plinth. Soft even neutral overcast daylight, no cast "
          "shadows, no sky, no people, no cars, no trees, no text, no signs, no logos.")


def walls_uz(w):
    return np.stack([w["base"][..., :2] @ w["e"], w["base"][..., 2]], -1)


def at(w, u, z):
    q = walls_uz(w)
    d = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])
    a = ((u - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (z - q[:, 0, 1])) / np.where(d == 0, 1, d)
    b = ((q[:, 1, 0] - q[:, 0, 0]) * (z - q[:, 0, 1]) - (u - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])) / np.where(d == 0, 1, d)
    k = np.flatnonzero((d != 0) & (a >= 0) & (b >= 0) & (a + b <= 1))
    if not len(k):
        return None
    t = w["base"][k[0]]
    n = np.cross(t[1] - t[0], t[2] - t[0])[:2]
    return t[0] + a[k[0]] * (t[1] - t[0]) + b[k[0]] * (t[2] - t[0]), n / (np.linalg.norm(n) or 1)


def quads(qs):
    return [np.array([q[0], q[i], q[i + 1]]) for q in qs for i in (1, 2)]


def box(w, ua, ub, za, zb, d, top=False):
    hit = at(w, (ua + ub) / 2, (za + zb) / 2)
    if hit is None:
        return []
    p, n = hit
    t = np.array([-n[1], n[0]])
    if t @ w["e"] < 0.7:
        return []
    q = lambda u, o, z: np.r_[p[:2] + (u - p[:2] @ w["e"]) / (t @ w["e"]) * t + o * n, z]
    qs = [[q(ua, d, za), q(ub, d, za), q(ub, d, zb), q(ua, d, zb)], [q(ua, 0, za), q(ua, d, za), q(ua, d, zb), q(ua, 0, zb)],
          [q(ub, d, za), q(ub, 0, za), q(ub, 0, zb), q(ub, d, zb)]]
    return quads(qs + ([[q(ua, 0, zb), q(ua, d, zb), q(ub, d, zb), q(ub, 0, zb)]] if top else []))


def parapet(w, h=0.6, d=0.25):
    out = []
    for t in w["base"]:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        n = np.r_[n[:2], 0] / (np.linalg.norm(n[:2]) or 1)
        for i in range(3):
            a, b, c = t[i], t[(i + 1) % 3], t[(i + 2) % 3]
            if abs(a[2] - b[2]) < 0.05 and a[2] > 2 and c[2] < a[2] - 0.3 and np.linalg.norm(b - a) > 0.3:
                a, b = (a, b) if np.cross(b - a, [0, 0, 1]) @ n > 0 else (b, a)
                up, ins = np.array([0, 0, h]), -d * n
                out += quads([[a, b, b + up, a + up], [b + ins, a + ins, a + ins + up, b + ins + up], [a + up, b + up, b + ins + up, a + ins + up]])
    return out


def plan(b, w, bal, entry):
    g, fh, width = b["style"][0], b["fh"], w["u1"] - w["u0"]
    nf = max(1, b["floors"] or round(b["h"] / 3))
    top, out = min(w["zt"], nf * fh), []
    bays = max(1, round(width / (6.0 if g in ("industry", "shed", "parking", "transit") else 3.2)))
    bw, nd = width / bays, max(1, round(width / (7 if g in ("row", "house") else 20)))
    door = set(np.round((np.arange(nd) + 0.5) * bays / nd - 0.5).astype(int)) if entry else set()
    for i in range(bays):
        u = w["u0"] + (i + 0.5) * bw
        for f in range(nf):
            z = f * fh
            if z + fh > top + 0.1:
                break
            if g in ("shop", "office", "public", "transit") and (f == 0 or g == "transit"):
                out.append(("glass", u - bw / 2 + 0.3, u + bw / 2 - 0.3, z + 0.3, z + fh - 0.4))
            elif g in ("industry", "shed", "parking"):
                if g == "parking" and f > 0 or f == nf - 1 and g != "parking":
                    out.append(("glass", u - bw / 2 + 0.6, u + bw / 2 - 0.6, z + fh - 1.6, z + fh - 0.5))
                if f == 0 and i == bays // 2 and entry:
                    out.append(("door", u - 2, u + 2, 0.2, min(4.2, fh - 0.4)))
            elif f == 0 and i in door:
                out += [("door", u - 0.9, u + 0.9, 0.2, 2.4), ("hood", u - 1.3, u + 1.3, 2.6, 2.85)]
            elif i in door and g == "apt":
                out.append(("win", u - 0.6, u + 0.6, z - fh / 2 + 0.6, z - fh / 2 + 1.8))
            elif bal and (f > 0 or g == "row") and i % (3 if g == "apt" else 2) == 1:
                out += [("dark", u - 1.7, u + 1.7, z + 0.1, z + fh - 0.3), ("rail", u - 1.7, u + 1.7, z + 0.1, z + 1.2)]
            elif width >= 6 and (bal or entry or i % 2 == 0 or g in ("row", "house")):
                out.append(("win", u - 0.8, u + 0.8, z + 0.8, z + 2.3))
    return out


def dress(b):
    ln = [w["u1"] - w["u0"] for w in b["walls"]]
    sw = [np.array([w["e"][1], -w["e"][0]]) @ [-0.6, -0.8] for w in b["walls"]]
    long = [k for k in range(len(ln)) if ln[k] >= 0.6 * max(ln)]
    bal = max(long, key=lambda k: sw[k]) if b["style"][0] in ("apt", "row") and max(ln) >= 12 else -1
    entry = min(long, key=lambda k: sw[k]) if len(long) > 1 or bal < 0 else -1
    r = b["roof"]
    n = np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]) if len(r) else np.zeros((0, 3))
    a = np.linalg.norm(n, axis=1)
    flat = len(r) and a[n[:, 2] > 0.97 * a].sum() > 0.8 * a.sum()
    for k, w in enumerate(b["walls"]):
        w["base"] = w["tris"]
        w["plan"] = plan(b, w, k == bal, k == entry)
        add = [t for kind, *c in w["plan"] if kind in ("rail", "hood") for t in box(w, *c, 1.0 if kind == "rail" else 1.4, kind == "hood")]
        w["cap"] = parapet(w) if flat else []
        w["tris"] = np.vstack([w["base"], *(np.array(x).reshape(-1, 3, 3) for x in (add, w["cap"]) if len(x))])
        u = w["tris"][..., :2] @ w["e"]
        w["u0"], w["u1"], w["z0"] = u.min(), u.max(), w["tris"][..., 2].min()
        w["zt"] = max(w["tris"][..., 2].max(), w["z0"] + 0.5)


def control(b, w, s):
    (W, H), col = size(w, s), SKIN[b["style"][2]]
    px = lambda u, z: ((u - w["u0"]) * s, (w["zt"] - z) * s)
    m, layer, img = Image.new("L", (W, H), 0), Image.new("RGB", (W, H), col), Image.new("RGB", (W, H), BG)
    for t in w["base"]:
        ImageDraw.Draw(m).polygon([px(p[:2] @ w["e"], p[2]) for p in t], fill=255)
    d = ImageDraw.Draw(layer)
    d.rectangle([0, px(0, 0.5)[1], W, H], fill=tuple(int(c * 0.7) for c in col))
    for z in np.arange(b["fh"], w["zt"], b["fh"]):
        d.line([(0, px(0, z)[1]), (W, px(0, z)[1])], fill=tuple(int(c * 0.82) for c in col), width=max(1, round(s * 0.08)))
    inside = lambda c: all(0 <= x < W and 0 <= y < H and m.getpixel((int(x), int(y))) for x, y in (px(c[0], c[2]), px(c[1], c[2]), px(c[0], c[3]), px(c[1], c[3])))
    for kind, *c in w["plan"]:
        if inside(c):
            (x0, y1), (x1, y0) = px(c[0], c[2]), px(c[1], c[3])
            if kind in ("win", "glass"):
                d.rectangle([x0, y0, x1, y1], fill=FRAME)
                d.rectangle([x0 + s * 0.08, y0 + s * 0.08, x1 - s * 0.08, y1 - s * 0.08], fill=GLASS)
                for x in np.arange(x0 + s * 1.5, x1 - s * 0.5, s * 1.5) if kind == "glass" else [(x0 + x1) / 2] if c[1] - c[0] > 1.8 else []:
                    d.line([(x, y0), (x, y1)], fill=FRAME, width=max(1, round(s * 0.06)))
            else:
                d.rectangle([x0, y0, x1, y1], fill={"door": DOOR, "hood": HOOD, "rail": RAIL, "dark": GLASS}[kind])
    img.paste(layer, mask=m)
    for t in w["cap"]:
        ImageDraw.Draw(img).polygon([px(p[:2] @ w["e"], p[2]) for p in t], fill=CAP)
    for r in w["rings"]:
        ImageDraw.Draw(img).line([px(p[:2] @ w["e"], p[2]) for p in np.vstack([r, r[:1]])], fill=(70, 70, 70), width=2)
    return img


def scale(w, s=PPM):
    return min(s, 1500 / max(w["u1"] - w["u0"], 1), 990 / max(w["zt"] - w["z0"], 1))


def layout(ws):
    fits = [(max(p[0] for p in pos) + 1, -s, s, pos) for s in range(PPM, 15, -1) if (pos := pack([size(w, scale(w, s)) for w in ws], 1536, 1024, 8))]
    return min(fits)[2:]


def chosen():
    bs, _ = buildings()
    bs = [b for b in bs if b["id"] in PICK]
    for b in bs:
        dress(b)
    return bs


def sheets(*only):
    jobs, book = [], {"sheets": []}
    for b in chosen():
        ws = list(enumerate(b["walls"]))
        s, pos = layout(b["walls"])
        for pg in range(max(p[0] for p in pos) + 1):
            base, its = Image.new("RGB", (1536, 1024), BG), []
            for (i, w), (p, x, y) in zip(ws, pos):
                if p == pg:
                    base.paste(control(b, w, scale(w, s)), (x, y))
                    its.append([f"{b['id']}:{i}", x, y, x + size(w, scale(w, s))[0], y + size(w, scale(w, s))[1], scale(w, s)])
            name = "fac_" + hashlib.md5(json.dumps(its).encode()).hexdigest()[:8]
            book["sheets"].append({"file": f"{name}.jpg", "items": its})
            text = PROMPT.format(n=len(its), what=WHAT[b["style"][0]], era=ERA[b["style"][1]], look=LOOK[b["style"][2]])
            jobs.append((name, base, text, sorted((RAW / "photos").glob(f"{b['style'][0]}_{b['style'][1]}*.jpg")) + [ROOT / "data/hype/hype_s.jpg"]))
    for j in jobs:
        j[1].save(RAW / f"{j[0]}_in.png")
    jobs = [j for j in jobs if not only or j[0].startswith(only)]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(call, jobs))
    (TEX / "facade.json").write_text(json.dumps(book))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
