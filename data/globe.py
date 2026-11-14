import sys
from math import cos, radians

import numpy as np
import shapefile
from PIL import Image, ImageDraw

W, H, S = 4096, 2048, 2
SEA, LAND, EDGE = (9, 10, 12), (38, 41, 44), (62, 66, 70)
JAM = {"Istanbul": 2.2, "Mumbai": 2.0, "Bogota": 2.2, "Manila": 2.1, "Bangalore": 2.2, "Moscow": 1.9, "Lima": 2.0, "Mexico City": 1.9, "Jakarta": 2.0, "Bangkok": 1.9,
       "London": 1.8, "Paris": 1.7, "Los Angeles": 1.8, "New York": 1.7, "Dublin": 1.9, "Bucharest": 1.9, "Delhi": 1.8, "Lagos": 2.1, "Cairo": 1.9, "São Paulo": 1.8,
       "Rio de Janeiro": 1.8, "Dhaka": 2.1, "Kolkata": 1.8, "Tehran": 1.8, "Rome": 1.7, "Milan": 1.6, "Nairobi": 1.8, "Santiago": 1.6, "Toronto": 1.6, "Sydney": 1.5}
RAMP = [(0.0, (41, 179, 87)), (0.42, (41, 179, 87)), (0.62, (255, 224, 38)), (0.8, (242, 26, 15)), (1.0, (128, 0, 107))]


def main(land, places, out):
    im = Image.new("RGB", (W * S, H * S), SEA)
    d = ImageDraw.Draw(im)
    for s in shapefile.Reader(land).shapes():
        ends = list(s.parts) + [len(s.points)]
        for a, b in zip(ends, ends[1:]):
            d.polygon([((x + 180) / 360 * W * S, (90 - y) / 180 * H * S) for x, y in s.points[a:b]], fill=LAND, outline=EDGE)
    base = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32)
    f = np.zeros((H, W), np.float32)
    for r in shapefile.Reader(places).records():
        m, lat, lon = r["pop_max"] / 1e6, r["latitude"], r["longitude"]
        if m < 0.03 or abs(lat) > 80:
            continue
        sy = (0.45 + 0.3 * min(m, 30) ** 0.5) * H / 180
        sx = sy / cos(radians(lat))
        cx, cy, ry, rx = (lon + 180) / 360 * W, (90 - lat) / 180 * H, int(3 * sy) + 1, int(3 * sx) + 1
        ys, xs = np.arange(int(cy) - ry, int(cy) + ry + 1), np.arange(int(cx) - rx, int(cx) + rx + 1)
        g = np.exp(-(ys[:, None] - cy) ** 2 / (2 * sy * sy) - (xs[None, :] - cx) ** 2 / (2 * sx * sx)) * 0.4 * m ** 0.5 * JAM.get(r["name"], 1.0)
        np.add.at(f, (np.clip(ys, 0, H - 1)[:, None], xs[None, :] % W), g)
    t = f / (f + 1.2)
    k, c = zip(*RAMP)
    heat = np.stack([np.interp(t, k, [v[i] for v in c]) for i in range(3)], -1)
    a = (np.clip((t - 0.03) / 0.09, 0, 1) * (0.6 + 0.35 * np.clip(t / 0.6, 0, 1)))[..., None]
    Image.fromarray((base * (1 - a) + heat * a).astype(np.uint8)).save(out, quality=90, method=6)


if __name__ == "__main__":
    main(*sys.argv[1:])
