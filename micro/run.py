import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sumo
import sumolib

from data.geo import PBF, VIZ, local, lonlat

HYPE = (24.6345, 60.1555)
R = 1500
WARM, REC = 600, 1200
OUT = Path(__file__).resolve().parent / "out"
TOOLS = Path(sumo.SUMO_HOME) / "tools"
TYPES = ["car", "van", "truck", "bus", "moto"]
SHARE = {"car": 0.83, "van": 0.10, "truck": 0.05, "moto": 0.02}
VTYPES = {
    "car": dict(vClass="passenger", length=(4.0, 4.6, 5.0), accel=2.6, decel=4.5, maxSpeed=50, sf="normc(1.05,0.10,0.8,1.2)", minGap=2.0, tau=1.2),
    "van": dict(vClass="delivery", length=(5.0, 5.9, 6.9), accel=2.0, decel=4.0, maxSpeed=38, sf="normc(1.0,0.08,0.8,1.2)", minGap=2.5, tau=1.3),
    "truck": dict(vClass="truck", length=(10.0, 12.0, 16.5), accel=1.0, decel=3.5, maxSpeed=25, sf="normc(0.95,0.05,0.8,1.05)", minGap=3.0, tau=1.6),
    "moto": dict(vClass="motorcycle", length=(2.0, 2.2, 2.4), accel=4.0, decel=6.0, maxSpeed=50, sf="normc(1.1,0.12,0.8,1.4)", minGap=1.5, tau=1.0),
    "bus": dict(vClass="bus", length=(12.0, 13.5, 18.0), accel=1.2, decel=4.0, maxSpeed=25, sf="normc(0.95,0.05,0.85,1.05)", minGap=2.5, tau=1.4),
}


def sh(*cmd):
    subprocess.run([str(c) for c in cmd], check=True, cwd=OUT, stdout=subprocess.DEVNULL)


def tool(name, *args):
    sh(sys.executable, TOOLS / name, *args)


def vtypes():
    root = ET.Element("additional")
    dist = {"mix": ET.SubElement(root, "vTypeDistribution", id="mix"), "bus": ET.SubElement(root, "vTypeDistribution", id="buses")}
    for k, p in VTYPES.items():
        for i, (L, w) in enumerate(zip(p["length"], (0.25, 0.5, 0.25))):
            a = dict(id=f"{k}{i}", vClass=p["vClass"], length=str(L), accel=str(p["accel"]), decel=str(p["decel"]),
                     emergencyDecel="9", maxSpeed=str(p["maxSpeed"]), speedFactor=p["sf"], minGap=str(p["minGap"]),
                     tau=str(p["tau"]), delta="4", sigma="0", carFollowModel="IDM", laneChangeModel="LC2013",
                     lcStrategic="1", lcCooperative="1", lcSpeedGain="1", lcKeepRight="1")
            ET.SubElement(dist.get(k, dist["mix"]), "vType", a, probability=str(SHARE.get(k, 1) * w))
    ET.ElementTree(root).write(OUT / "types.add.xml")


def network():
    x0, y0 = local(*HYPE)
    (w, s), (e, n) = lonlat(x0 - R, y0 - R), lonlat(x0 + R, y0 + R)
    sh("osmium", "extract", "-O", "-s", "complete_ways", "-b", f"{w},{s},{e},{n}", PBF, "-o", "focus.osm")
    sh(Path(sumo.SUMO_HOME) / "bin" / "netconvert", "--osm-files", "focus.osm", "-o", "net.xml",
       "--type-files", Path(sumo.SUMO_HOME) / "data/typemap/osmNetconvert.typ.xml",
       "--keep-edges.by-vclass", "passenger,bus", "--remove-edges.isolated", "--geometry.remove",
       "--ramps.guess", "--junctions.join", "--tls.guess-signals", "--tls.join", "--tls.default-type", "actuated",
       "--roundabouts.guess", "--ptstop-output", "stops.xml", "--ptline-output", "lines.xml",
       "--ptline-clean-up", "--no-turnarounds.except-deadend")


def scale_from_log(path, year):
    years = json.loads(Path(path).read_text())["years"]
    links = json.loads((VIZ / "net.json").read_text())["links"]
    x0, y0 = local(*HYPE)
    near = [i for i, l in enumerate(links) if min(np.hypot(gx - x0, gy - y0) for gx, gy in l["geom"]) < R]
    return sum(years[year]["flow"][i] for i in near) / max(sum(years[0]["flow"][i] for i in near), 1)


def demand(scale):
    end = WARM + REC
    tool("randomTrips.py", "-n", "net.xml", "-o", "trips.xml", "-r", "cars.rou.xml", "-e", end,
         "-p", 3600 / (4200 * scale), "--fringe-factor", "20", "--lanes", "--speed-exponent", "6",
         "--min-distance", "600",
         "--trip-attributes", 'type="mix" departLane="best" departSpeed="max"', "--seed", "7",
         "--additional-file", "types.add.xml", "--vclass", "passenger", "--validate", "--remove-loops")
    tool("ptlines2flows.py", "-n", "net.xml", "-s", "stops.xml", "-l", "lines.xml", "-o", "bus.rou.xml",
         "-p", 600 / scale, "-e", end, "--types", "bus", "--use-osm-routes", "--extend-to-fringe",
         "--no-vtypes", "--ignore-errors", "--random-begin", "--seed", "7", "--min-stops", "2")
    t = ET.parse(OUT / "bus.rou.xml")
    for f in list(t.getroot().iter("flow")):
        f.set("type", "buses")
        if f.get("line").split(":")[0].endswith("N"):
            t.getroot().remove(f)
    t.write(OUT / "bus.rou.xml")


def simulate():
    sh(Path(sumo.SUMO_HOME) / "bin" / "sumo", "-n", "net.xml", "-a", "types.add.xml,stops.xml",
       "-r", "trips.xml,bus.rou.xml", "-e", WARM + REC, "--step-length", "1", "--fcd-output", "fcd.xml",
       "--fcd-output.attributes", "x,y,type", "--device.fcd.begin", WARM, "--time-to-teleport", "120",
       "--device.rerouting.probability", "0.5", "--device.rerouting.period", "60",
       "--ignore-route-errors", "--no-step-log", "--no-warnings", "--seed", "7", "--tripinfo-output", "trips.out.xml")


def export():
    net = sumolib.net.readNet(str(OUT / "net.xml"))
    tracks, kind = {}, {}
    for _, el in ET.iterparse(OUT / "fcd.xml"):
        if el.tag == "timestep":
            k = round(float(el.get("time"))) - WARM
            for v in el:
                tracks.setdefault(v.get("id"), []).append((k, float(v.get("x")), float(v.get("y"))))
                kind[v.get("id")] = next(i for i, t in enumerate(TYPES) if v.get("type").startswith(t))
            el.clear()
    veh, pts = [], []
    off = 0
    for vid, tr in tracks.items():
        a = np.array(tr)
        cut = np.flatnonzero(np.diff(a[:, 0]) != 1) + 1
        for seg in np.split(a, cut):
            if len(seg) < 2:
                continue
            lon, lat = net.convertXY2LonLat(seg[:, 1], seg[:, 2])
            x, y = local(lon, lat)
            pts.append(np.round(np.c_[x, y] * 10).astype("<i4"))
            veh.append([kind[vid], int(seg[0, 0]), off, len(seg)])
            off += len(seg)
    VIZ.mkdir(parents=True, exist_ok=True)
    np.concatenate(pts).tofile(VIZ / "cars.bin")
    (VIZ / "cars.json").write_text(json.dumps({"dt": 1.0, "types": TYPES, "veh": veh}, separators=(",", ":")))
    return net, veh


def report(net, veh):
    v = np.array(veh)
    trips = [t.attrib for t in ET.parse(OUT / "trips.out.xml").getroot()]
    spd = [float(t["routeLength"]) / float(t["duration"]) * 3.6 for t in trips if float(t["duration"]) > 0]
    print(f"tracks {len(v)}  points {v[:, 3].sum()}  tls {len(net.getTrafficLights())}  edges {len(net.getEdges())}")
    print("share", {t: round(float((v[:, 0] == i).mean()), 3) for i, t in enumerate(TYPES)})
    print(f"mean trip speed {np.mean(spd):.1f} km/h  bin {(VIZ / 'cars.bin').stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--log")
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--reuse-net", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    scale = a.scale * (scale_from_log(a.log, a.year) if a.log else 1)
    vtypes()
    if not a.reuse_net or not (OUT / "net.xml").exists():
        network()
    demand(scale)
    simulate()
    report(*export())


if __name__ == "__main__":
    main()
