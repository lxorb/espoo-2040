import hashlib
import json
import sys
from multiprocessing import Pool

import numpy as np
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union
from data.district import BG, DET, ROOT, TEX, buildings, call, glb, normals, prep, prism, save, tilebs
from data.facade import LOOKS, PROMPT, at, cls, dress, parapet, roles, unfold

CL = TEX / "clean"
ROOFS = {"flat-dark": "dark grey bituminous roofing membrane laid in long parallel strips with faint welded seams",
         "flat-light": "light grey PVC roofing membrane laid in long parallel strips with faint welded seams",
         "gravel": "flat roof covered with fine light grey gravel ballast",
         "green": "flat sedum green roof with mixed green, olive and reddish sedum plants",
         "tile-red": "red clay roof tiles in regular courses, the courses running horizontally across the image",
         "tile-brown": "dark brown concrete roof tiles in regular courses, the courses running horizontally across the image",
         "tile-black": "black glazed roof tiles in regular courses, the courses running horizontally across the image",
         "shingle": "dark grey asphalt shingles in regular courses, the courses running horizontally across the image",
         "metal-grey": "grey standing seam metal roofing, the seams running vertically from top to bottom of the image",
         "metal-green": "dark green standing seam metal roofing, the seams running vertically from top to bottom of the image",
         "metal-red": "dark red standing seam metal roofing, the seams running vertically from top to bottom of the image",
         "industrial": "light grey profiled metal sheet roofing with ribs running vertically and one translucent polycarbonate skylight strip running vertically"}
TILE = ("The first image is only a placeholder. Produce a seamless tileable texture that fills the whole image edge to edge: a perfectly top-down "
        "orthographic view of {}, as a crisp, clean architectural illustration texture in the style of the reference image. Uniform soft overcast "
        "light, no shadows, no perspective, no edges, no borders, no objects, no text.")
HOUSES = {"wood-red": LOOKS["wood"][0], "wood-yellow": LOOKS["wood"][1], "wood-grey": LOOKS["wood"][2],
          "wood-white": ((242, 240, 232), "white painted vertical timber boards"), "wood-brown": ((104, 74, 58), "dark brown stained vertical timber boards"),
          "brick-red": LOOKS["brick"][0], "render-white": LOOKS["render"][0], "wood-blue": ((176, 198, 214), "pale blue painted vertical timber boards"),
          "render-beige": LOOKS["render"][1]}


def house(col):
    img, s = Image.new("RGB", (1536, 1024), BG), 60
    d = ImageDraw.Draw(img)
    for row, (y, wins, door) in enumerate([(60, [1.6, 6.4], 4.0), (400, [2.0, 6.0], None), (740, [4.0], None)]):
        d.rectangle([48, y, 48 + 8 * s, y + 3 * s], fill=col, outline=(70, 70, 70), width=2)
        d.rectangle([48, y + 2.6 * s, 48 + 8 * s, y + 3 * s], fill=tuple(int(c * 0.6) for c in col)) if row < 2 else None
        for u in wins:
            ww, z0, z1 = (1.4, 0.9, 2.3) if row < 2 else (0.9, 1.0, 2.0)
            d.rectangle([48 + (u - ww / 2) * s, y + (3 - z1) * s, 48 + (u + ww / 2) * s, y + (3 - z0) * s], fill=(236, 236, 232))
            d.rectangle([48 + (u - ww / 2 + 0.08) * s, y + (3 - z1 + 0.08) * s, 48 + (u + ww / 2 - 0.08) * s, y + (3 - z0 - 0.08) * s], fill=(52, 66, 86))
        if door:
            d.rectangle([48 + (door - 0.5) * s, y + 0.6 * s, 48 + (door + 0.5) * s, y + 2.75 * s], fill=(96, 64, 44))
    return img


def seamless(img, n=512):
    a = np.asarray(img.crop((256, 0, 1280, 1024)).resize((n, n), Image.LANCZOS), float)
    r = np.roll(a, (n // 2, n // 2), (0, 1))
    g = np.abs(np.linspace(-1, 1, n))
    m = np.clip((np.maximum(g[:, None], g[None]) - 0.55) / 0.35, 0, 1)[..., None]
    return Image.fromarray((a * (1 - m) + r * m).astype(np.uint8))


def textures():
    CL.mkdir(exist_ok=True)
    ref = sorted(TEX.glob("roof_*.jpg"))[:2]
    jobs = [(f"clean_roof_{k}", Image.new("RGB", (1536, 1024), (128, 128, 128)), TILE.format(v), ref, True) for k, v in ROOFS.items()]
    jobs += [(f"clean_house_{k}", house(c), PROMPT.format(n=3, what="detached family houses", era="built between the 1960s and the 2000s", look=desc,
                                                         rail="no balconies") + " Each strip is one storey of a small Finnish wooden or masonry detached house.",
              [ROOT / "data/hype/hype_s.jpg"]) for k, (c, desc) in HOUSES.items()]
    res = {j[0]: call(j) for j in jobs}
    for k in ROOFS:
        seamless(res[f"clean_roof_{k}"]).save(CL / f"roof-{k}.webp", quality=82)
    for k in HOUSES:
        for row, y in enumerate((60, 400, 740)):
            res[f"clean_house_{k}"].crop((48, y, 48 + 480, y + (150 if row == 2 else 180))).resize((480, 180)).save(CL / f"house-{k}-{'gwm'[row]}.webp", quality=82)


def vary(path):
    a = np.asarray(Image.open(path).convert("RGB"), float)
    dark = a.mean(-1) < 90
    tint = lambda x, k: np.where(dark[..., None] if x is a else dark[:, ::-1, None], x * k, x)
    Image.fromarray(np.clip(np.hstack([a, tint(a[:, ::-1], (0.9, 0.97, 1.1)), tint(a, (1.12, 1.05, 0.94)), a[:, ::-1]]), 0, 255).astype(np.uint8)).save(path, quality=84)


def modules():
    items = {it[0]: (sh["file"], it[1:5], it[5]) for sh in json.loads((TEX / "facade.json").read_text())["sheets"] for it in sh["items"]}
    fine, lib = {k.split(":")[0] for k in items}, []
    for b in buildings()[0]:
        if b["id"] not in fine:
            continue
        dress(b)
        rec, fh = {"id": b["id"], "cls": cls(b), "mat": "metal" if b["look"][2] and b["h"] < 15 else b["style"][2], "era": b["style"][1]}, b["fh"]

        def crop(i, w, ua, ub, za, zb, name):
            if name in rec or any(at(w, u, z) is None for u in (ua + 0.1, ub - 0.1) for z in (za + 0.1, zb - 0.1)):
                return
            f, (x0, y0, _, _), s = items[f"{b['id']}:{i}"]
            Image.open(TEX / f).crop((round(x0 + (ua - w["u0"]) * s), round(y0 + (w["zt"] - zb) * s), round(x0 + (ub - w["u0"]) * s),
                                      round(y0 + (w["zt"] - za) * s))).save(CL / f"mod-{b['id']}-{name}.webp", quality=84)
            rec[name] = round(ub - ua, 2)

        ws = sorted(enumerate(b["walls"]), key=lambda iw: -(iw[1]["u1"] - iw[1]["u0"]))
        for i, w in ws:
            if w["bw"] < 1:
                continue
            bw, nb, u0, top = w["bw"], round((w["u1"] - w["u0"]) / w["bw"]), w["u0"], w["base"][..., 2].max()
            if rec["cls"] == "big":
                for h in (top, top * 0.85, top * 0.7):
                    crop(i, w, u0 + 1, u0 + 1 + min(14, w["u1"] - u0 - 2), 0, h, "w")
                continue
            f = 1 if w["nf"] >= 3 else max(0, w["nf"] - 1)
            for i0 in range(max(0, nb - 2) if f else 0):
                if not {i0, i0 + 1, i0 + 2} & w["door"] and (w["role"] != "bal" or i0 % 3 == 0):
                    crop(i, w, u0 + i0 * bw, u0 + (i0 + 3) * bw, f * fh, (f + 1) * fh, "b" if w["role"] == "bal" else "m")
            for d in sorted(w["door"]):
                if 2 <= d <= nb - 4:
                    crop(i, w, u0 + (d - 2) * bw, u0 + (d + 4) * bw, 0, fh, "g")
            if w["role"] != "gable" and nb >= 3:
                crop(i, w, u0, u0 + 3 * bw, 0, fh, "g")
            if w["role"] == "gable" and w["u1"] - u0 > 4:
                for k in range(f, -1, -1):
                    crop(i, w, u0, w["u1"], k * fh, (k + 1) * fh, "e")
        for r in "mb":
            if r in rec:
                vary(CL / f"mod-{b['id']}-{r}.webp")
                rec[r] *= 4
        lib.append(rec)
        print(rec, flush=True)
    (CL / "mod.json").write_text(json.dumps(lib))


DETC = DET.parent / "detail-clean"
FLAT = {"res": ["flat-dark", "flat-light", "gravel"], "com": ["flat-dark", "flat-light", "gravel"], "big": ["industrial", "flat-light"]}
PITCH = {"res": ["tile-red", "tile-brown", "tile-black", "shingle", "metal-grey", "metal-green", "metal-red"], "com": ["metal-grey", "tile-black", "metal-red"],
         "big": ["industrial", "metal-grey"]}
HOUSE = ["wood-red", "wood-yellow", "wood-white", "wood-grey", "wood-blue", "wood-brown"] * 2 + ["wood-red", "wood-white", "render-white", "render-beige",
                                                                                                "render-white", "render-beige", "brick-red", "brick-red"]
HROOF = ["tile-red", "tile-black", "tile-brown", "metal-grey", "tile-red", "tile-black", "shingle", "metal-red", "metal-grey", "metal-green"]
SPRITE = {"vent": (0.9, 0.9), "ac": (2.4, 1.4), "sky": (1.8, 1.2), "solar": (3.4, 3.0)}


def items():
    img = Image.new("RGB", (512, 128), (70, 72, 76))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, 124, 124], fill=(150, 152, 156), outline=(60, 60, 62), width=4)
    d.ellipse([34, 34, 94, 94], fill=(110, 112, 116), outline=(60, 60, 62), width=4)
    d.rectangle([132, 4, 252, 124], fill=(214, 216, 216), outline=(90, 92, 94), width=4)
    for c in (162, 222):
        d.ellipse([c - 26, 38, c + 26, 90], fill=(120, 124, 128), outline=(70, 70, 72), width=3)
        d.line([(c - 22, 64), (c + 22, 64)], fill=(70, 70, 72), width=3)
        d.line([(c, 42), (c, 86)], fill=(70, 70, 72), width=3)
    d.rectangle([260, 4, 380, 124], fill=(200, 204, 206))
    d.rectangle([272, 16, 368, 112], fill=(96, 128, 150))
    d.line([(320, 16), (320, 112)], fill=(200, 204, 206), width=4)
    d.rectangle([388, 4, 508, 124], fill=(176, 180, 184))
    for x in range(2):
        for y in range(3):
            d.rectangle([392 + x * 58, 8 + y * 39, 446 + x * 58, 43 + y * 39], fill=(28, 38, 70))
            d.line([(392 + x * 58 + 27, 8 + y * 39), (392 + x * 58 + 27, 43 + y * 39)], fill=(70, 84, 120), width=1)
    img.save(CL / "items.webp", quality=90)


def hsh(b, k=""):
    return int(hashlib.md5((b["id"] + k).encode()).hexdigest()[:8], 16)


def chop(t, u, z):
    lo, hi = ([], []), ([], [])
    for tri, uu in zip(t, u):
        p = np.c_[tri, uu]
        for side, keep in ((lo, lambda q: q <= z), (hi, lambda q: q >= z)):
            poly = []
            for a, c in zip(p, np.roll(p, -1, 0)):
                if keep(a[2]):
                    poly.append(a)
                if keep(a[2]) != keep(c[2]) and a[2] != c[2]:
                    poly.append(a + (c - a) * (z - a[2]) / (c[2] - a[2]))
            for k in range(1, len(poly) - 1):
                q = np.array([poly[0], poly[k], poly[k + 1]])
                side[0].append(q[:, :3])
                side[1].append(q[:, 3])
    return [(np.array(s[0]).reshape(-1, 3, 3), np.array(s[1]).reshape(-1, 3)) for s in (lo, hi)]


def tidy(b, small):
    r = b["roof"]
    if not len(r) or b["foot"].is_empty:
        return
    n, a = normals(r)
    p = n[:, 2] < 0.97
    messy = len(r) > 6 + b["foot"].area / 15
    if p.sum():
        h = np.bincount(((np.arctan2(n[p, 1], n[p, 0]) + np.pi) / (np.pi / 4)).astype(int) % 8, a[p], 8)
        messy |= max(max(h[k] + h[k + 4] for k in range(4)), max(h[k::2].sum() for k in range(2))) < 0.75 * a[p].sum()
    if not messy:
        return
    f = max(getattr(b["foot"], "geoms", [b["foot"]]), key=lambda q: q.area).simplify(0.3)
    if small:
        m = f.minimum_rotated_rectangle
        b["faces"] = prism((m if f.area > 0.7 * m.area else f).exterior.coords, float(r[..., 2].max()), True)
    else:
        z = r[..., 2].mean(1)
        o = np.argsort(z)
        b["faces"], b["parapet"] = prism(f.exterior.coords, float(z[o][np.searchsorted(np.cumsum(a[o]), a.sum() / 2)])), True
    prep(b)


def dressup(b, lib, out):
    c, g = cls(b), b["style"][0]
    c = "com" if c == "big" and b["h"] > 12 else c
    mat = "metal" if c == "big" and b["h"] < 15 else b["style"][2]
    small = g in ("house", "row", "shed") and b["foot"].area < 400 and b["h"] < 9
    tidy(b, small)
    if b["foot"].is_empty:
        for w in b["walls"]:
            w["u"], w["n"] = w["tris"][..., :2] @ w["e"], np.array([w["e"][1], -w["e"][0]])
    else:
        unfold(b)
    pick = lambda opts, k: opts[hsh(b, k) % len(opts)]
    if small:
        hk = pick(HOUSE, "h")
        rows, wid = {r: f"house-{hk}-{r}" for r in "gwm"}, {r: 8.0 for r in "gwmeb"}
        rows["e"], rows["b"] = rows["w"], rows["m"]
    else:
        rec = pick(lib.get((c, mat)) or lib.get((c, None)) or lib[(None, None)], "m")
        have = [r for r in "mbegw" if r in rec]
        rows = {r: f"mod-{rec['id']}-{r if r in rec else have[0]}" for r in "mbegw"}
        wid = {r: rec[r if r in rec else have[0]] for r in "mbegw"}
    fh, top = b["fh"], max(b["h"], 3)
    if b.get("parapet"):
        for w in b["walls"]:
            w["base"] = w["tris"]
            t, u = (np.array(x) for x in parapet(w))
            if len(t):
                out("roof-metal-grey", t.reshape(-1, 3, 3), np.stack([u.reshape(-1, 3) / 4, t.reshape(-1, 3, 3)[..., 2] / 4], -1))
    for w, role in zip(b["walls"], roles(b) if b["walls"] else []):
        if c == "big" and not small and "w" in rec:
            out(rows["w"], w["tris"], np.stack([w["u"] / wid["w"], 1 - w["tris"][..., 2] / top], -1))
            continue
        (lt, lu), (ht, hu) = chop(w["tris"], w["u"], fh)
        gr = "e" if role == "gable" else "g" if not small or role == "entry" else "w"
        up = "e" if role == "gable" else "b" if role == "bal" else "m"
        out(rows[gr], lt, np.stack([lu / wid[gr], (fh - lt[..., 2]) / fh], -1))
        out(rows[up], ht, np.stack([hu / wid[up], -ht[..., 2] / fh], -1))
    r = b["roof"]
    if not len(r):
        return
    n = normals(r)[0]
    fl = n[:, 2] > 0.97
    flat = pick(FLAT[c] + (["green"] if b["style"][1] == "new" and c == "res" and b["foot"].area < 2500 else []), "f")
    out(f"roof-{flat}", r[fl], r[fl][..., :2] * (1, -1) / (12 if flat == "industrial" else 8))
    if (~fl).any():
        s = n[~fl, :2] / (np.linalg.norm(n[~fl, :2], axis=1, keepdims=True) + 1e-9)
        t = r[~fl]
        uv = np.stack([np.einsum("tvk,tk->tv", t[..., :2], np.c_[-s[:, 1], s[:, 0]]) / 4,
                       np.einsum("tvk,tk->tv", t[..., :2], s) / (4 * n[~fl, 2:3])], -1)
        out(f"roof-{pick(HROOF if small else PITCH[c], 'p')}", t, uv)
    if fl.any():
        decals(b, r[fl], c, out)


def decals(b, r, c, out):
    poly = unary_union([Polygon(t[:, :2]).buffer(0.05) for t in r]).buffer(-1.6)
    if poly.is_empty or poly.area < 20:
        return
    rng = np.random.default_rng(hsh(b, "d"))
    m = np.array(b["foot"].minimum_rotated_rectangle.exterior.coords) if not b["foot"].is_empty else np.array([[0, 0], [1, 0], [1, 1]])
    eu = max((m[1] - m[0], m[2] - m[1]), key=np.linalg.norm)
    eu = eu / (np.linalg.norm(eu) or 1)
    ev = np.array([-eu[1], eu[0]])
    kinds = ["vent"] * 4 + ["ac"] * (3 if c != "res" else 1) + ["sky"] * (2 if c != "res" else 1) + ["solar"] * (3 if b["style"][1] == "new" else 0)
    x0, y0, x1, y1 = poly.bounds
    p = rng.uniform((x0, y0), (x1, y1), (400, 2))
    p = p[shapely.contains_xy(poly, p[:, 0], p[:, 1])][: int(min(14 if c == "res" else 80, poly.area / 90)) + 1]
    tris, uvs = [], []
    for q in p:
        k = kinds[rng.integers(len(kinds))]
        i = list(SPRITE).index(k)
        a, h = np.array(SPRITE[k]) / 2
        tri = r[np.argmin([np.linalg.norm(t[:, :2].mean(0) - q) for t in r])]
        z = tri[:, 2].max() + 0.06
        cs = [np.r_[q + sa * a * eu + sb * h * ev, z] for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        us = [((i + 0.04 + 0.92 * (sa > 0)) / 4, 0.04 + 0.92 * (sb < 0)) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        tris += [[cs[0], cs[1], cs[2]], [cs[0], cs[2], cs[3]]]
        uvs += [[us[0], us[1], us[2]], [us[0], us[2], us[3]]]
    if tris:
        out("items", np.array(tris), np.array(uvs))


def model(bs, prefix="tex-"):
    lib, acc = {}, {}
    for rec in (r for r in json.loads((CL / "mod.json").read_text()) if set(r) & set("mbegw")):
        for key in ((rec["cls"], rec["mat"]), (rec["cls"], None), (None, None)):
            lib.setdefault(key, []).append(rec)
    out = lambda name, v, uv: acc.setdefault(name, ([], []))[0].append(v.reshape(-1, 3)) or acc[name][1].append(uv.reshape(-1, 2)) if len(v) else None
    for b in bs:
        dressup(b, lib, out)
    return glb([(f"{prefix}{k}.webp", np.vstack(v), np.vstack(uv)) for k, (v, uv) in sorted(acc.items())]) if acc else None


def tile(t):
    bs = tilebs(t)
    if data := model(bs):
        save(DETC / f"{t[0]}_{t[1]}.glb", data)
    return t, len(bs), len(data or b"")


def tiles(procs="4", *names):
    DETC.mkdir(exist_ok=True)
    items()
    for f in CL.glob("*.webp"):
        (DETC / f"tex-{f.name}").write_bytes(f.read_bytes())
    done = json.loads((DET.parent / "detail.json").read_text())
    todo = [tuple(x) for x in done["tiles"] if not names or f"{x[0]}_{x[1]}" in names]
    with Pool(int(procs)) as p:
        for r in p.imap_unordered(tile, todo):
            print(*r, flush=True)
    have = sorted(tuple(map(int, f.stem.split("_"))) for f in DETC.glob("*.glb"))
    (DET.parent / "detail-clean.json").write_text(json.dumps({"size": done["size"], "tiles": have, "partial": [x for x in done["partial"] if tuple(x) in have]}, separators=(",", ":")))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
