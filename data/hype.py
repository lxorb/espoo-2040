import numpy as np
import trimesh
from PIL import Image
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon
from data.geo import ROOT

GLB = ROOT / "viz" / "assets" / "hype.glb"
TEX = ROOT / "data" / "hype"

HYPE_SW, HYPE_SE, HYPE_NW = (-1540.8, 1212.7), (-1495.8, 1203.2), (-1527.5, 1275.3)
PARKING = [(-1494.8, 1202.9), (-1481.3, 1266.8), (-1445.7, 1259.3), (-1457.5, 1203.8), (-1467.7, 1197.1)]

W, D = 46.0, 64.0
G, L2, L3, TOP, NTOP = 6.3, 15.7, 25.3, 30.0, 31.5
P_LEVELS = [3.5 + 3.2 * i for i in range(8)]
P_TOP = 27.0

MATS = {
    "concrete": (0.92, 0.92, 0.90, 0.0, 0.85),
    "glass": (0.36, 0.48, 0.56, 0.6, 0.15),
    "wood": (0.70, 0.36, 0.17, 0.0, 0.7),
    "metal": (0.55, 0.57, 0.60, 0.7, 0.4),
    "louvre": (0.60, 0.61, 0.62, 0.6, 0.5),
    "pglass": (0.66, 0.82, 0.78, 0.4, 0.2),
    "deck": (0.80, 0.64, 0.45, 0.0, 0.8),
    "water": (0.30, 0.74, 0.84, 0.2, 0.1),
    "solar": (0.12, 0.16, 0.28, 0.5, 0.3),
    "sign": (1.0, 1.0, 1.0, 0.0, 0.5),
    "dark": (0.36, 0.37, 0.38, 0.0, 0.9),
    "roof": (0.55, 0.56, 0.56, 0.0, 0.95),
}

parts = []


def frame(o, a, b):
    o = np.array(o)
    eu = np.subtract(a, o) / np.linalg.norm(np.subtract(a, o))
    ev = np.array([-eu[1], eu[0]])
    if np.dot(ev, np.subtract(b, o)) < 0:
        ev = -ev
    return o, eu, ev


H = frame(HYPE_SW, HYPE_SE, HYPE_NW)
P = frame(PARKING[0], tuple(np.add(PARKING[0], np.subtract(PARKING[2], PARKING[1]))), PARKING[1])
PW = float(np.dot(np.subtract(PARKING[2], P[0]), P[1]))
PD = float(np.dot(np.subtract(PARKING[1], P[0]), P[2]))
SKINS = {"hype": (H, -1, W + 1, -1, D + 1, 35.0), "parking": (P, -1, PW + 1, -1, PD + 1, 31.0)}
SIDES = "snwet"
INSET = {"hype": 3.0, "parking": 10.0}


def rect(fr, u0, u1, v0, v1):
    o, eu, ev = fr
    return [tuple(o + u * eu + v * ev) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]


def prism(poly, z0, z1, mat, cap=True):
    p = Polygon(poly)
    if not p.exterior.is_ccw:
        p = Polygon(list(p.exterior.coords)[::-1])
    ring = np.array(p.exterior.coords)[:-1]
    verts, uvs, faces = [], [], []
    run = 0.0
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        L = float(np.linalg.norm(b - a))
        n = len(verts)
        verts += [(*a, z0), (*b, z0), (*b, z1), (*a, z1)]
        uvs += [(run, z0), (run + L, z0), (run + L, z1), (run, z1)]
        faces += [(n, n + 1, n + 2), (n, n + 2, n + 3)]
        run += L
    if cap:
        for t in constrained_delaunay_triangles(p).geoms:
            tri = np.array(t.exterior.coords)[:3]
            a, b = tri[1] - tri[0], tri[2] - tri[0]
            if a[0] * b[1] - a[1] * b[0] < 0:
                tri = tri[::-1]
            for z, order in ((z1, (0, 1, 2)), (z0, (0, 2, 1))):
                n = len(verts)
                verts += [(*tri[k], z) for k in order]
                uvs += [tuple(tri[k]) for k in order]
                faces.append((n, n + 1, n + 2))
    parts.append((mat, np.array(verts), np.array(uvs), np.array(faces)))


def box(u0, u1, v0, v1, z0, z1, mat, fr=H):
    prism(rect(fr, u0, u1, v0, v1), z0, z1, mat)


def face(side, a0, a1, z0, z1, mat, d=0.12, at=None):
    if side == "S":
        box(a0, a1, (at or 0) - d, at or 0, z0, z1, mat)
    elif side == "N":
        box(a0, a1, at or D, (at or D) + d, z0, z1, mat)
    elif side == "W":
        box((at or 0) - d, at or 0, a0, a1, z0, z1, mat)
    else:
        box(at or W, (at or W) + d, a0, a1, z0, z1, mat)


def hype():
    box(1.5, W, 1.5, D, 0, G, "concrete")
    box(0, 1.5, 18, D, 0, G, "concrete")
    box(0, 0.9, 0, 0.9, 0, G, "concrete")
    box(0, W, 0, D, G, L3, "concrete")
    box(0, W, 0, 20, L3, TOP, "concrete")
    box(24, W, 20, 44, L3, TOP, "concrete")
    box(0, W, 44, D, L3, NTOP, "concrete")

    for a0, a1, m in ((1.5, 5, "glass"), (5, 8.5, "wood"), (8.5, 15, "glass"), (15, 21, "wood"),
                      (21, 31, "glass"), (31, 34, "wood"), (34, 40, "glass"), (40, 44, "wood")):
        face("S", a0, a1, 0.2, G - 0.6, m, at=1.5)
    for a0, a1, m in ((1.5, 9, "glass"), (9, 14, "wood"), (14, 17.5, "glass")):
        face("W", a0, a1, 0.2, G - 0.6, m, at=1.5)
    for a0, a1 in ((24, 30), (40, 46)):
        face("W", a0, a1, 0.8, G - 1.0, "glass")
    for z in (G, L2, L3):
        for s in "SNWE":
            face(s, -0.15, (W if s in "SN" else D) + 0.15, z - 0.25, z + 0.15, "metal", d=0.2)
    face("S", 0, 22, G + 0.4, L2 - 0.3, "glass")
    face("S", 0, 13, L2 + 0.3, L3 - 0.3, "glass")
    face("S", 33.5, 34.5, G + 0.5, L3 - 0.5, "glass")
    face("S", 0, 40, L3 + 0.3, TOP - 0.3, "glass")
    face("W", 0, 10, G + 0.4, L3 - 0.3, "glass")
    face("W", 22, 40, G + 3.2, L2 - 2.2, "glass")
    face("W", 0, 20, L3 + 0.3, TOP - 0.3, "glass")
    face("W", 48, 49, G + 0.5, L3 - 0.5, "glass")
    face("N", 14, 15, G + 0.5, L3 - 0.5, "glass")
    face("N", 31, 32, G + 0.5, L3 - 0.5, "glass")
    face("N", 6, 40, L3 + 0.5, NTOP - 0.6, "louvre", d=0.3)
    face("E", 20, 44, L3 + 0.6, TOP - 0.6, "louvre", d=0.3)
    for s, a0, a1, z0 in (("S", 3, 12, L3 - 3.6), ("W", 50, 60, NTOP - 3.4), ("N", 0.8, 5.8, NTOP - 3.0),
                          ("N", 40.5, 45.5, NTOP - 3.0), ("E", 49, 59, NTOP - 3.2)):
        face(s, a0, a1, z0, z0 + 2.2, "sign", d=0.25)
    for u0, u1, v0, v1, z1 in ((0, W, 0, 20, TOP), (24, W, 20, 44, TOP), (0, W, 44, D, NTOP)):
        for a, b, c, d in ((u0, u1, v0, v0 + 0.3), (u0, u1, v1 - 0.3, v1), (u0, u0 + 0.3, v0, v1), (u1 - 0.3, u1, v0, v1)):
            box(a, b, c, d, z1, z1 + 0.5, "metal")
    box(0, 24, 20, 44, L3, L3 + 0.3, "deck")
    box(2.5, 7.5, 23, 40, L3 + 0.3, L3 + 0.35, "water")
    box(11, 17, 36.5, 43.5, L3 + 0.3, L3 + 3.3, "wood")
    box(11, 17, 21, 26, L3 + 0.3, L3 + 3.0, "wood")
    face("W", 20, 44, L3 + 0.3, L3 + 1.6, "glass", d=0.05)
    box(18, 30, 50, 58, NTOP, NTOP + 2.6, "louvre")
    box(4, 12, 4, 14, TOP, TOP + 1.8, "louvre")


def parking():
    body = Polygon(PARKING)
    prism(list(body.buffer(-0.3, join_style=2).exterior.coords)[:-1], 0, P_LEVELS[0], "pglass")
    prism(PARKING, P_LEVELS[0], P_LEVELS[-1], "concrete")
    ring = np.array(PARKING)
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        n = np.array([b[1] - a[1], a[0] - b[0]]) / np.linalg.norm(b - a) * 0.3
        prism([tuple(a), tuple(b), tuple(b + n), tuple(a + n)], P_LEVELS[-1], P_TOP, "concrete")
    for z in P_LEVELS:
        prism(list(body.buffer(0.08, join_style=2).exterior.coords)[:-1], z - 0.12, z + 0.12, "metal", cap=False)
    prism(list(body.buffer(-0.2, join_style=2).exterior.coords)[:-1], P_LEVELS[-1], P_LEVELS[-1] + 0.05, "roof")
    width = PW
    box(width - 0.2, width + 0.1, 54, 61, 0, 2.6, "dark", P)
    box(width - 0.2, width + 0.1, 54, 61, 2.6, P_LEVELS[0], "concrete", P)
    box(width - 6.5, width - 1.5, 10, 50, P_LEVELS[-1], P_LEVELS[-1] + 3.2, "concrete", P)
    box(width - 6.3, width - 1.7, 10.2, 49.8, P_LEVELS[-1] + 3.2, P_LEVELS[-1] + 3.3, "glass", P)
    for i in range(5):
        box(3, width - 9, 6 + i * 10.5, 12 + i * 10.5, P_LEVELS[-1] + 0.3, P_LEVELS[-1] + 1.1, "solar", P)


def merge(chunks):
    off = np.cumsum([0] + [len(c[1]) for c in chunks])[:-1]
    return (np.vstack([c[1] for c in chunks]), np.vstack([c[2] for c in chunks]),
            np.vstack([c[3] + k for c, k in zip(chunks, off)]))


def local_frame(fr, v):
    xy = v[:, :2] - fr[0]
    return xy @ fr[1], xy @ fr[2], v[:, 2]


def side_uv(side, u, v, z, u0, u1, v0, v1, zt):
    x = {"s": (u - u0) / (u1 - u0), "n": (u1 - u) / (u1 - u0), "w": (v1 - v) / (v1 - v0),
         "e": (v - v0) / (v1 - v0), "t": (u - u0) / (u1 - u0)}[side]
    return np.c_[x, (v - v0) / (v1 - v0) if side == "t" else z / zt]


def keys(group, chunks, tri, skin):
    mats = np.concatenate([[c[0]] * len(c[3]) for c in chunks])
    if not skin or group not in SKINS or not all((TEX / f"{group}_{s}.jpg").exists() for s in SIDES):
        return mats
    fr, u0, u1, v0, v1, zt = SKINS[group]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nu, nv = n[:, :2] @ fr[1], n[:, :2] @ fr[2]
    k = np.argmax(np.abs(np.c_[nu, nv, n[:, 2]]), axis=1)
    side = np.where(k == 2, "t", np.where(k == 0, np.where(nu > 0, "e", "w"), np.where(nv > 0, "n", "s")))
    cu, cv, _ = local_frame(fr, tri.mean(1))
    d = INSET[group] + 1
    outer = {"s": cv < v0 + d, "n": cv > v1 - d, "w": cu < u0 + d, "e": cu > u1 - d}
    ok = np.select([side == s for s in "snwe"], [outer[s] for s in "snwe"], True)
    return np.where(ok, side, mats)


def build(skin=True):
    parts.clear()
    groups, scene = {}, trimesh.Scene()
    for group, fn in (("hype", hype), ("parking", parking)):
        n = len(parts)
        fn()
        groups[group] = chunks = parts[n:]
        tri, tuv = (lambda v, uv, f: (v[f], uv[f]))(*merge(chunks))
        key = keys(group, chunks, tri, skin)
        for name in dict.fromkeys(key):
            t = tri[key == name].reshape(-1, 3)
            if name in MATS:
                r, g, b, metal, rough = MATS[name]
                mat = trimesh.visual.material.PBRMaterial(name=name, baseColorFactor=[r, g, b, 1.0],
                                                          metallicFactor=metal, roughnessFactor=rough)
                tex = tuv[key == name].reshape(-1, 2)
            else:
                img = Image.open(TEX / f"{group}_{name}.jpg")
                mat = trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=img, metallicFactor=0.0)
                tex = side_uv(name, *local_frame(SKINS[group][0], t), *SKINS[group][1:])
            m = trimesh.Trimesh(np.c_[t[:, 0], t[:, 2], -t[:, 1]], np.arange(len(t)).reshape(-1, 3), process=False,
                                visual=trimesh.visual.TextureVisuals(uv=tex, material=mat))
            scene.add_geometry(m, node_name=f"{group}_{name}", geom_name=f"{group}_{name}")
    return scene, groups


if __name__ == "__main__":
    s, _ = build()
    GLB.parent.mkdir(parents=True, exist_ok=True)
    GLB.write_bytes(s.export(file_type="glb"))
    print(GLB, sum(len(g.faces) for g in s.geometry.values()), "tris", GLB.stat().st_size, "bytes")
