import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from data.district import BG, call
from data.hype import MATS, SIDES, SKINS, TEX, local_frame, side_uv, build

PPM = 24
PHOTOS = {
    "hype_s": ["1564-ulkokuva-crisp2jpg-e1755686086271-1400x984.jpg", "hype_refe1.1-3.jpg"],
    "hype_w": ["hype_refe1.1-3.jpg", "1564-ulkokuva2-1000x838.jpg"],
    "hype_n": ["dsc9273-hdr-e1763543272165-scaled-1400x809.jpg"],
    "hype_e": ["dsc9273-hdr-e1763543272165-scaled-1400x809.jpg"],
    "hype_t": ["1564-terassiallasnakyma-1400x934.jpg", "kattosauna-nettisivuille-1400x460.jpg"],
    "parking_s": ["1564-ulkokuva-crisp2jpg-e1755686086271-1400x984.jpg"],
    "parking_e": ["dsc9273-hdr-e1763543272165-scaled-1400x809.jpg"],
    "parking_n": ["dsc9273-hdr-e1763543272165-scaled-1400x809.jpg"],
    "parking_w": ["dsc9273-hdr-e1763543272165-scaled-1400x809.jpg"],
    "parking_t": [],
}
VIEW = {"s": "south facade", "n": "north facade", "w": "west facade", "e": "east facade", "t": "roof seen from directly above"}
PROMPT = (
    "The first image is an exact orthographic {view} of {what}, drawn in flat colour codes. Repaint it as a realistic "
    "architectural illustration texture of that same {view}, perfectly straight-on with no perspective. Keep every outline, "
    "opening, floor line, panel boundary and sign exactly where it is: do not move, add, remove or resize anything, and keep the "
    "flat pale blue-grey background outside the building untouched and flat. Colour codes: near white = white ribbed precast "
    "concrete panels with fine vertical ribs and a staggered grid of large rectangular panel joints; dark blue-grey = curtain wall "
    "glazing with slim aluminium mullions and transoms and dim interior; orange-brown = umber glazed concrete panels with large "
    "curved line pattern; thin mid grey bands = aluminium floor-line flashings and parapet caps; pure white boxes = the white "
    "Hype logo sign: render only the white hexagonal H mark and the white word Hype mounted directly on the wall, never a white panel; light grey striped bands = vertical aluminium louvres; pale green "
    "= translucent green-tinted glass; light wood = timber deck; turquoise = swimming pool water; dark navy = solar panels; "
    "darker grey = roof membrane or service doors; on the roof, orange-brown boxes are dark timber sauna cabins. Soft even neutral overcast daylight, no cast shadows, no sky, no people, no cars, "
    "no trees. The other images are real photographs of the building for material reference only."
)
NOTES = {
    "hype_s": "The white box upper left marks the Hype logo on the glass. Ground storey: storefront glazing with doors alternating with umber panels.",
    "hype_w": "Left is north, right is south. Upper left block carries the Hype logo. The top storey middle is a roof terrace: a low frameless glass balustrade in front, dark timber sauna cabins behind it and a white concrete wall beyond. The top storey on the right is a fully glazed curtain wall, not solar panels. Grey boxes on top are rooftop air handling units with metal grilles. The middle band has one long horizontal ribbon window.",
    "hype_n": "The top band is grey vertical aluminium louvres between two white concrete end panels with Hype letters directly on the concrete. Grey box on top is a rooftop air handling unit.",
    "hype_e": "All white areas are white ribbed concrete, including the whole ground storey, which has no umber panels. The top band middle is grey vertical aluminium louvres. Grey boxes on top are rooftop air handling units with metal grilles.",
    "hype_t": "Roof plan. Light areas are pale grey roof membrane with welded seams and drains, not facade panels. The tan area is a timber deck terrace of light wooden boards, with a turquoise pool and two dark timber sauna cabin roofs. Grey rectangles are rooftop air handling units with fan grilles, not solar panels. Thin grey lines are aluminium parapet caps.",
    "parking_s": "Multi-storey car park: pale green translucent glass ground storey, closed white ribbed concrete panels above with a horizontal joint at each floor slab. No signs, logos or lettering anywhere.",
    "parking_n": "Multi-storey car park: pale green translucent glass ground storey, closed white ribbed concrete panels above with a horizontal joint at each floor slab. No signs, logos or lettering anywhere.",
    "parking_w": "Multi-storey car park: pale green translucent glass ground storey, closed white ribbed concrete panels above with a horizontal joint at each floor slab. No signs, logos or lettering anywhere.",
    "parking_e": "Multi-storey car park: pale green translucent glass ground storey with a dark vehicle entrance door, closed white ribbed concrete panels above with a horizontal joint at each floor slab. No signs, logos or lettering anywhere.",
    "parking_t": "Roof deck of the car park: grey asphalt parking deck with white parking bay lines, five rows of dark solar panels, and along the right edge a long ramp housing whose roof is a glass skylight with mullions, not water.",
}
WHAT = {"hype": "the Hybridiarena Hype sports arena in Kivenlahti, Espoo", "parking": "the Kivenlahti park and ride garage next to Hype"}


def raster(tris, cols, xy, depth, w, h):
    img = np.full((h, w, 3), BG, float)
    zb = np.full((h, w), np.inf)
    for t, c, p, d in zip(tris, cols, xy, depth):
        x0, x1 = max(0, int(p[:, 0].min())), min(w - 1, int(np.ceil(p[:, 0].max())))
        y0, y1 = max(0, int(p[:, 1].min())), min(h - 1, int(np.ceil(p[:, 1].max())))
        if x0 > x1 or y0 > y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        den = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(den) < 1e-9:
            continue
        a = ((gx - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (gy - p[0, 1])) / den
        b = ((p[1, 0] - p[0, 0]) * (gy - p[0, 1]) - (gx - p[0, 0]) * (p[1, 1] - p[0, 1])) / den
        m = (a >= -1e-6) & (b >= -1e-6) & (a + b <= 1 + 1e-6)
        z = d[0] + a * (d[1] - d[0]) + b * (d[2] - d[0])
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        m &= z < sub
        sub[m] = z[m]
        img[y0:y1 + 1, x0:x1 + 1][m] = c
    return img


def elevation(group, side, chunks):
    fr, u0, u1, v0, v1, zt = SKINS[group]
    span = {"s": u1 - u0, "n": u1 - u0, "w": v1 - v0, "e": v1 - v0, "t": u1 - u0}[side]
    tall = v1 - v0 if side == "t" else zt
    w, h = round(span * PPM), round(tall * PPM)
    tris, cols = [], []
    for mat, v, f in chunks:
        tris.append(v[f])
        cols += [np.array(MATS[mat][:3]) * 255] * len(f)
    tris = np.vstack(tris)
    u, vv, z = (a.reshape(-1, 3) for a in local_frame(fr, tris.reshape(-1, 3)))
    uv = side_uv(side, u.ravel(), vv.ravel(), z.ravel(), u0, u1, v0, v1, zt).reshape(-1, 3, 2)
    xy = np.stack([uv[..., 0] * w, (1 - uv[..., 1]) * h], -1)
    depth = {"s": vv, "n": -vv, "w": u, "e": -u, "t": -z}[side]
    img = raster(tris, cols, xy, depth, w, h)
    edge = np.asarray(Image.fromarray(img.astype(np.uint8)).convert("L").filter(ImageFilter.FIND_EDGES)) > 12
    img[edge] *= 0.45
    return Image.fromarray(img.astype(np.uint8))


def canvas(img):
    w, h = img.size
    tw, th = (1536, 1024) if w >= h else (1024, 1536)
    s = min(tw / w, th / h)
    nw, nh = round(w * s), round(h * s)
    box = ((tw - nw) // 2, (th - nh) // 2, (tw - nw) // 2 + nw, (th - nh) // 2 + nh)
    out = Image.new("RGB", (tw, th), BG)
    out.paste(img.resize((nw, nh), Image.LANCZOS), box[:2])
    return out, box


if __name__ == "__main__":
    ref, groups = Path(sys.argv[1]), build(skin=False)[1]
    for name in sys.argv[2:] or [f"{g}_{s}" for g in SKINS for s in SIDES]:
        group, side = name.split("_")
        base = elevation(group, side, groups[group])
        img, box = canvas(base)
        text = PROMPT.format(view=VIEW[side], what=WHAT[group]) + " " + NOTES[name]
        call((f"hypetex_{name}", img, text, [ref / p for p in PHOTOS[name]], True)).crop(box).resize(base.size, Image.LANCZOS).save(TEX / f"{name}.jpg", quality=88)
