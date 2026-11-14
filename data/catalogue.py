import json
import math
import osmium
from geo import PBF, ROOT, local

HW = {"motorway", "trunk", "primary", "secondary", "tertiary"}
LINKS = {h + "_link" for h in HW}
ROADS = {}


class Roads(osmium.SimpleHandler):
    def way(self, w):
        hw = w.tags.get("highway")
        if hw in HW or hw in LINKS:
            ROADS[w.id] = (w.tags.get("name", ""), hw in LINKS, [local(n.lon, n.lat) for n in w.nodes], w.tags.get("oneway") == "yes")


def ways(names, x0, y0, x1, y1, links=False):
    return sorted(i for i, (n, lk, g, _) in ROADS.items()
                  if (n in names or (links and lk)) and any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in g))


def near(names, x, y, r=350):
    return ways(names, x - r, y - r, x + r, y + r, links=True)


def gate(names, x, y, r=200):
    return ways(names, x - r, y - r, x + r, y + r)


def km(g):
    return sum(math.dist(a, b) for a, b in zip(g, g[1:])) / 1000


def roadkm(ids):
    return sum(km(ROADS[i][2]) / (2 if ROADS[i][3] else 1) for i in ids)


def new(g, lanes=2, kmh=80, cls="trunk", bridge=False):
    return {"geom": [[float(x), float(y)] for x, y in g], "lanes": lanes, "kmh": kmh, "cls": cls, "oneway": False, "bridge": bridge}


def newcost(parts, junctions=0):
    return round(sum(km(p["geom"]) * (15 if p["lanes"] > 1 else 5) * (4 if p["bridge"] else 1) for p in parts) + 25 * junctions)


def bus(pts, kmh, headway, was=None):
    n = 2 * km(pts) / kmh * 60 * (1 / headway - (1 / was if was else 0))
    return round(n * 0.45, 1)


L1, L2, T1, KE1, KE2, KE3 = "Länsiväylä", "Turunväylä", "Turuntie", "Kehä I", "Kehä II", "Kehä III"
ST = {"Leppavaara": (8866, 7847), "Kilo": (7137, 7744), "Kera": (5650, 7642), "Kauniainen": (4190, 7163),
      "Koivuhovi": (2658, 6682), "Tuomarila": (1516, 6617), "Espoo": (91, 6555), "Kauklahti": (-3059, 4926),
      "Kivenlahti": (-1201, 1072), "Espoonlahti": (-205, 360), "Finnoo": (2981, 656), "Matinkyla": (4518, 1342), "Niittykumpu": (5884, 2486), "Urheilupuisto": (6902, 2919),
      "Tapiola": (8164, 2929), "Aalto": (9294, 3944), "Keilaniemi": (9590, 2951), "Koivusaari": (10893, 1562),
      "Myyrmaki": (11303, 12434), "Martinlaakso": (11252, 14313), "Masala": (-6582, 1625), "Juvanmalmi": (4800, 12900),
      "Mankkaa": (6000, 4500), "Olari": (4300, 2900), "Suurpelto": (5000, 3900), "Lahnus": (6500, 18800)}


def stops(*names):
    return [list(ST[n]) for n in names]


def catalogue():
    k2n = [new([(5310, 8660), (5260, 9400), (5320, 10300), (5150, 11300), (4950, 12200), (4820, 12880)])]
    evm = [new([(480, 5420), (900, 4700), (1400, 3950), (2000, 3330), (2560, 2960)], 2, 60, "primary")]
    kib = [new([(-1760, 2170), (-2700, 2700), (-3600, 2950), (-4200, 2985)], 1, 60, "secondary"),
           new([(-4200, 2985), (-4800, 2930), (-5420, 2830)], 1, 60, "secondary", True),
           new([(-5420, 2830), (-6080, 2740)], 1, 60, "secondary")]
    man = [new([(5090, 4520), (5800, 4600), (6400, 4560), (6900, 4520)], 1, 50, "tertiary")]
    ola = [new([(2560, 2960), (3300, 3260), (4300, 3380), (5040, 3780)], 1, 50, "tertiary")]
    k3s = [i for i in ways([KE3], -7200, -900, -4000, 3800) if not ROADS[i][3]]
    k3n = [i for i in ways([KE3], -7200, 3800, -900, 7700) if not ROADS[i][3] and i not in k3s]
    kv = ways(["Kauklahdenväylä"], -6000, 1000, 0, 5500)
    ev = ways(["Espoonväylä", "Finnoontie"], -1500, 3700, 1000, 7500)
    l2w = ways([L2], -4500, 7400, 1500, 9300)
    l2e = ways([L2], 1500, 5800, 9300, 7000)
    l1w = ways([L1], -6900, -800, -1300, 1600)
    l1m = ways([L1], -1300, 600, 4600, 2100)
    k1 = ways([KE1], 8300, 3000, 9800, 7600)
    k2 = ways([KE2], 4800, 1900, 5600, 8700)
    vih = ways(["Vihdintie"], 1500, 12500, 8800, 21000)
    vt1 = ways([L2], 9300, 5800, 11500, 7000)
    return [
        {"id": "keha2-north", "name": "Kehä II extension Turuntie to Kehä III", "kind": "road",
         "cost": newcost(k2n, 2), "years": 4, "opex": 0.3, "new": k2n,
         "works": [{"osm": near([T1, KE2], 5310, 8660, 500) + near([KE3], 4820, 12880, 500), "cap": 0.8}]},
        {"id": "espoonvayla-mid", "name": "Espoonväylä central section", "kind": "road",
         "cost": newcost(evm, 1), "years": 3, "opex": 0.2, "new": evm,
         "works": [{"osm": near(["Finnoontie", "Espoonväylä", "Finnoonväylä", "Puolarintie"], 480, 5420, 600), "cap": 0.8}]},
        {"id": "kivenlahti-bridge", "name": "Kivenlahti to Kehä III bay bridge", "kind": "road",
         "cost": newcost(kib), "years": 3, "opex": 0.2, "new": kib,
         "works": [{"osm": near(["Kauklahdenväylä"], -1760, 2170, 300), "cap": 0.8}]},
        {"id": "mankkaa-link", "name": "Mankkaa link Kehä II to Vanhan-Mankkaan tie", "kind": "road",
         "cost": newcost(man), "years": 2, "opex": 0.05, "new": man},
        {"id": "olari-link", "name": "Puolarintie to Suurpelto link", "kind": "road",
         "cost": newcost(ola), "years": 2, "opex": 0.05, "new": ola},
        {"id": "keha3-south-2x2", "name": "Kehä III dual carriageway Länsiväylä to Kauklahti", "kind": "lanes",
         "cost": round(roadkm(k3s) * 12 + 30), "years": 4, "opex": 0.3,
         "mod": [{"osm": k3s, "lanes": 1, "kmh": 80, "cap": 1.1}], "works": [{"osm": k3s, "cap": 0.6}]},
        {"id": "keha3-north-2x2", "name": "Kehä III dual carriageway Kauklahti to Espoon keskus", "kind": "lanes",
         "cost": round(roadkm(k3n) * 12 + 25), "years": 3, "opex": 0.3,
         "mod": [{"osm": k3n, "lanes": 1, "kmh": 80, "cap": 1.1}], "works": [{"osm": k3n, "cap": 0.6}]},
        {"id": "kauklahdenvayla-2x2", "name": "Kauklahdenväylä widening to 2+2", "kind": "lanes",
         "cost": round(roadkm(kv) * 8), "years": 2, "opex": 0.1,
         "mod": [{"osm": kv, "lanes": 1, "cap": 1.0}], "works": [{"osm": kv, "cap": 0.7}]},
        {"id": "espoonvayla-2x2", "name": "Espoonväylä and Finnoontie widening to 2+2", "kind": "lanes",
         "cost": round(roadkm(ev) * 8), "years": 2, "opex": 0.1,
         "mod": [{"osm": ev, "lanes": 1, "cap": 1.0}], "works": [{"osm": ev, "cap": 0.7}]},
        {"id": "turunvayla-west-lanes", "name": "Turunväylä third lanes Espoon keskus to Histan", "kind": "lanes",
         "cost": round(roadkm(l2w) * 10), "years": 3, "opex": 0.2,
         "mod": [{"osm": l2w, "lanes": 1, "cap": 1.0}], "works": [{"osm": l2w, "cap": 0.75}]},
        {"id": "turunvayla-east-lanes", "name": "Turunväylä fourth lanes Kehä II to Kehä I", "kind": "lanes",
         "cost": round(roadkm(l2e) * 12), "years": 3, "opex": 0.2,
         "mod": [{"osm": l2e, "lanes": 1, "cap": 1.0}], "works": [{"osm": l2e, "cap": 0.75}]},
        {"id": "lansivayla-west-lanes", "name": "Länsiväylä third lanes Kehä III to Kivenlahti", "kind": "lanes",
         "cost": round(roadkm(l1w) * 12 + 20), "years": 3, "opex": 0.2,
         "mod": [{"osm": l1w, "lanes": 1, "cap": 1.0}], "works": [{"osm": l1w, "cap": 0.7}]},
        {"id": "vihdintie-2x2", "name": "Vihdintie widening Kehä III to Lahnus", "kind": "lanes",
         "cost": round(roadkm(vih) * 8), "years": 3, "opex": 0.2,
         "mod": [{"osm": vih, "lanes": 1, "kmh": 80, "cap": 1.0}], "works": [{"osm": vih, "cap": 0.7}]},
        {"id": "maarinsolmu", "name": "Maarinsolmu interchange on Kehä I", "kind": "junction",
         "cost": 33.7, "years": 3, "opex": 0.1,
         "src": "https://valtioneuvosto.fi/-/1410829/hallitus-esittaa-taydennyksia-vuoden-2025-maararahoihin-ja-edistaa-liikennehankkeita-yli-miljardilla-eurolla",
         "mod": [{"osm": near([KE1, "Kalevalantie", "Maarintie"], 8640, 4250, 400), "cap": 1.35}],
         "works": [{"osm": near([KE1], 8640, 4250, 700), "cap": 0.6}]},
        {"id": "keilaniemi-junction", "name": "Kehä I and Länsiväylä junction at Keilaniemi", "kind": "junction",
         "cost": 60, "years": 3, "opex": 0.1,
         "mod": [{"osm": near([KE1, L1], 9300, 2400, 450), "cap": 1.25}],
         "works": [{"osm": near([KE1, L1], 9300, 2400, 900), "cap": 0.65}]},
        {"id": "keha2-lansivayla-junction", "name": "Kehä II and Länsiväylä junction upgrade", "kind": "junction",
         "cost": 45, "years": 3, "opex": 0.1,
         "mod": [{"osm": near([KE2, L1], 5200, 2000, 450), "cap": 1.25}],
         "works": [{"osm": near([KE2, L1], 5200, 2000, 800), "cap": 0.65}]},
        {"id": "keha2-turunvayla-junction", "name": "Kehä II and Turunväylä junction area", "kind": "junction",
         "cost": 12, "years": 2, "opex": 0.05,
         "src": "https://espoo.oncloudos.com/kokous/20231048-15.PDF",
         "mod": [{"osm": near([KE2, L2, "Nihtisilta"], 5150, 6350, 450), "cap": 1.2}],
         "works": [{"osm": near([KE2, L2], 5150, 6350, 700), "cap": 0.75}]},
        {"id": "kivenlahti-junction", "name": "Kivenlahti interchange rebuild", "kind": "junction",
         "cost": 30, "years": 2, "opex": 0.05,
         "mod": [{"osm": near([L1, "Kauklahdenväylä", "Vanha Jorvaksentie"], -1700, 1250, 400), "cap": 1.3}],
         "works": [{"osm": near([L1, "Kauklahdenväylä"], -1700, 1250, 700), "cap": 0.7}]},
        {"id": "keha3-kauklahti-junction", "name": "Kehä III and Kauklahdenväylä junction signals", "kind": "junction",
         "cost": 2, "years": 1, "opex": 0.05,
         "mod": [{"osm": near([KE3, "Kauklahdenväylä"], -3940, 4750, 300), "cap": 1.15}]},
        {"id": "finnoo-roundabouts", "name": "Finnoontie and Nöykkiönkatu roundabouts", "kind": "junction",
         "cost": 4, "years": 1, "opex": 0.02,
         "mod": [{"osm": near(["Finnoontie", "Nöykkiönkatu"], 1050, 3700, 300) + near(["Finnoontie", "Puolarintie", "Finnoonväylä"], 2000, 2950, 300), "cap": 1.15}]},
        {"id": "buslane-lansivayla", "name": "Länsiväylä bus lanes Espoonlahti to Matinkylä", "kind": "buslane",
         "cost": round(roadkm(l1m) * 0.6, 1), "years": 1, "opex": bus(stops("Espoonlahti", "Matinkyla", "Koivusaari"), 45, 6),
         "mod": [{"osm": l1m, "lanes": -1}], "pt": [{"stops": stops("Espoonlahti", "Finnoo", "Matinkyla", "Koivusaari"), "kmh": 45, "headway": 6}]},
        {"id": "buslane-turunvayla", "name": "Turunväylä bus lanes Kehä II to Helsinki", "kind": "buslane",
         "cost": round(roadkm(l2e + vt1) * 0.6, 1), "years": 1, "opex": bus(stops("Kera", "Leppavaara"), 50, 5) + 2,
         "mod": [{"osm": l2e + vt1, "lanes": -1}], "pt": [{"stops": [[5200, 6300], [9300, 6700], [11500, 6000]], "kmh": 50, "headway": 5}]},
        {"id": "buslane-keha1", "name": "Kehä I bus lanes Keilaniemi to Leppävaara", "kind": "buslane",
         "cost": round(roadkm(k1) * 0.6, 1), "years": 1, "opex": 0.3,
         "mod": [{"osm": k1, "lanes": -1}], "pt": [{"stops": stops("Keilaniemi", "Aalto", "Leppavaara"), "kmh": 30, "headway": 5}]},
        {"id": "buslane-keha2", "name": "Kehä II bus lanes Matinkylä to Kilo", "kind": "buslane",
         "cost": round(roadkm(k2) * 0.6, 1), "years": 1, "opex": bus(stops("Matinkyla", "Suurpelto", "Kilo"), 35, 7.5),
         "mod": [{"osm": k2, "lanes": -1}], "pt": [{"stops": stops("Matinkyla", "Olari", "Suurpelto", "Mankkaa", "Kera", "Kilo", "Leppavaara"), "kmh": 35, "headway": 7.5}]},
        {"id": "espoo-city-rail", "name": "Espoo city rail Leppävaara to Kauklahti", "kind": "rail",
         "cost": 275, "years": 6, "opex": 6, "src": "https://www.kauniainen.fi/wp-content/uploads/2023/07/vaylavirasto_espoon-kaupunkirata_mediatilaisuuden-esitys_15.6.2022.pdf",
         "pt": [{"stops": stops("Leppavaara", "Kilo", "Kera", "Kauniainen", "Koivuhovi", "Tuomarila", "Espoo", "Kauklahti"), "kmh": 45, "headway": 5}],
         "works": [{"osm": near([T1, "Kauniaistentie", "Espoontie"], 4190, 7163, 500), "cap": 0.8}]},
        {"id": "lansirata", "name": "Länsirata first phase", "kind": "rail",
         "cost": 1300, "years": 8, "opex": 10, "src": "https://yle.fi/a/74-20192608",
         "pt": [{"stops": [list(ST["Espoo"]), [-4300, 10900], [-11000, 14300]], "kmh": 90, "headway": 15}],
         "works": [{"osm": ways([L2], -4500, 8500, -500, 9500), "cap": 0.85}]},
        {"id": "metro-west", "name": "Länsimetro extension Kivenlahti to Masala", "kind": "rail",
         "cost": round(km(stops("Kivenlahti", "Masala")) * 1.2 * 150), "years": 7, "opex": 8,
         "pt": [{"stops": [list(ST["Kivenlahti"]), [-4300, 600], list(ST["Masala"])], "kmh": 45, "headway": 4}],
         "works": [{"osm": near([L1, "Vanha Jorvaksentie"], -1201, 1072, 400), "cap": 0.8}]},
        {"id": "lrt-espoonlahti-keskus", "name": "Light rail Espoonlahti to Espoon keskus", "kind": "rail",
         "cost": round(km(stops("Espoonlahti", "Kivenlahti", "Kauklahti", "Espoo")) * 1.15 * 25), "years": 4, "opex": 8,
         "pt": [{"stops": [list(ST["Espoonlahti"]), list(ST["Kivenlahti"]), [-1760, 2170], [-2500, 3600], list(ST["Kauklahti"]), [-1500, 5800], list(ST["Espoo"])], "kmh": 25, "headway": 7.5}],
         "works": [{"osm": kv, "cap": 0.7}]},
        {"id": "jokeri-west", "name": "Raide-Jokeri extension Keilaniemi to Matinkylä", "kind": "rail",
         "cost": round(km(stops("Keilaniemi", "Tapiola", "Urheilupuisto", "Niittykumpu", "Matinkyla")) * 1.2 * 25), "years": 4, "opex": 5,
         "pt": [{"stops": stops("Keilaniemi", "Tapiola", "Urheilupuisto", "Niittykumpu", "Matinkyla"), "kmh": 22, "headway": 5}],
         "works": [{"osm": ways(["Tapiolantie", "Merituulentie", "Kalastajantie", "Piispansilta"], 4400, 1300, 9400, 3200), "cap": 0.75}]},
        {"id": "bus-520", "name": "Trunk bus 520 every 5 minutes", "kind": "bus",
         "cost": 2, "years": 1, "opex": bus(stops("Matinkyla", "Leppavaara", "Martinlaakso"), 25, 5, 10),
         "pt": [{"stops": stops("Matinkyla", "Niittykumpu", "Mankkaa", "Kilo", "Leppavaara", "Myyrmaki", "Martinlaakso"), "kmh": 25, "headway": 5}]},
        {"id": "bus-530", "name": "Trunk bus 530 every 5 minutes", "kind": "bus",
         "cost": 2, "years": 1, "opex": bus(stops("Matinkyla", "Espoo", "Juvanmalmi", "Myyrmaki"), 28, 5, 10),
         "pt": [{"stops": stops("Matinkyla", "Olari", "Suurpelto", "Espoo", "Juvanmalmi", "Myyrmaki"), "kmh": 28, "headway": 5}]},
        {"id": "bus-kivenlahti-trunk", "name": "New trunk bus Kivenlahti to Espoon keskus", "kind": "bus",
         "cost": 3, "years": 1, "opex": bus(stops("Kivenlahti", "Kauklahti", "Espoo"), 28, 7.5),
         "pt": [{"stops": [list(ST["Espoonlahti"]), list(ST["Kivenlahti"]), [-1760, 2170], list(ST["Kauklahti"]), list(ST["Espoo"])], "kmh": 28, "headway": 7.5}]},
        {"id": "bus-north-espoo", "name": "Trunk bus Lahnus to Leppävaara", "kind": "bus",
         "cost": 2, "years": 1, "opex": bus(stops("Lahnus", "Juvanmalmi", "Leppavaara"), 35, 10),
         "pt": [{"stops": [list(ST["Lahnus"]), [5600, 16000], list(ST["Juvanmalmi"]), [5300, 9500], list(ST["Kilo"]), list(ST["Leppavaara"])], "kmh": 35, "headway": 10}]},
        {"id": "toll-helsinki-cordon", "name": "Congestion charge at the Helsinki border", "kind": "toll",
         "cost": 60, "years": 2, "opex": 8,
         "toll": {"osm": gate([L1], 10300, 1800) + gate([L2], 10600, 6200) + gate([T1], 10600, 8200) + gate(["Vihdintie"], 9800, 11500) + gate([KE1], 10200, 8900), "eur": [1, 2, 4]}},
        {"id": "toll-lansivayla", "name": "Länsiväylä corridor charge", "kind": "toll",
         "cost": 15, "years": 1, "opex": 2,
         "toll": {"osm": gate([L1], 7000, 2200), "eur": [1, 2, 4]}},
        {"id": "toll-turunvayla", "name": "Turunväylä corridor charge", "kind": "toll",
         "cost": 15, "years": 1, "opex": 2,
         "toll": {"osm": gate([L2], 3000, 5950), "eur": [1, 2, 4]}},
        {"id": "toll-keha1", "name": "Kehä I corridor charge", "kind": "toll",
         "cost": 15, "years": 1, "opex": 2,
         "toll": {"osm": gate([KE1], 8550, 5500), "eur": [1, 2, 4]}},
        {"id": "parking-tapiola", "name": "Tapiola and Otaniemi parking charge", "kind": "parking",
         "cost": 1.5, "years": 1, "opex": 0.4, "park": {"area": [[7600, 2300], [9900, 2300], [9900, 4400], [7600, 4400]], "eur": [2, 5]}},
        {"id": "parking-leppavaara", "name": "Leppävaara parking charge", "kind": "parking",
         "cost": 1, "years": 1, "opex": 0.3, "park": {"area": [[8000, 7200], [9600, 7200], [9600, 8400], [8000, 8400]], "eur": [2, 5]}},
        {"id": "parking-matinkyla", "name": "Matinkylä parking charge", "kind": "parking",
         "cost": 1, "years": 1, "opex": 0.3, "park": {"area": [[3900, 900], [5300, 900], [5300, 2000], [3900, 2000]], "eur": [2, 5]}},
        {"id": "parking-espoonlahti", "name": "Espoonlahti and Kivenlahti parking charge", "kind": "parking",
         "cost": 1, "years": 1, "opex": 0.3, "park": {"area": [[-1900, -100], [500, -100], [500, 1500], [-1900, 1500]], "eur": [2, 5]}},
        {"id": "pnr-kivenlahti", "name": "Kivenlahti metro park and ride", "kind": "pnr",
         "cost": 15, "years": 2, "opex": 0.3, "pnr": {"at": [-1350, 1200], "spaces": 500}},
        {"id": "pnr-matinkyla", "name": "Matinkylä metro park and ride", "kind": "pnr",
         "cost": 15, "years": 2, "opex": 0.3, "pnr": {"at": [4700, 1500], "spaces": 500}},
        {"id": "pnr-kauklahti", "name": "Kauklahti station park and ride", "kind": "pnr",
         "cost": 6, "years": 1, "opex": 0.1, "pnr": {"at": [-3150, 4850], "spaces": 300}},
        {"id": "pnr-espoo", "name": "Espoon keskus station park and ride", "kind": "pnr",
         "cost": 12, "years": 2, "opex": 0.2, "pnr": {"at": [200, 6450], "spaces": 400}},
    ]


if __name__ == "__main__":
    Roads().apply_file(str(PBF), locations=True)
    cat = catalogue()
    (ROOT / "data" / "catalogue.json").write_text(json.dumps(cat, ensure_ascii=False, indent=1))
    for p in cat:
        print(f"{p['id']:28s} {p['kind']:9s} {p['cost']:8.1f} {p['years']} {sum(len(m['osm']) for m in p.get('mod', []) + [p.get('toll', {'osm': []})])}")
    print(len(cat), round(sum(p["cost"] for p in cat)))
