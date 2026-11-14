from pathlib import Path
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "out"
VIZ = ROOT / "viz" / "data"
PBF = RAW / "espoo.osm.pbf"
ORIGIN = (370000.0, 6670000.0)
BBOX = (24.40, 60.05, 25.00, 60.42)

_fwd = Transformer.from_crs(4326, 3067, always_xy=True)


def local(lon, lat):
    e, n = _fwd.transform(lon, lat)
    return e - ORIGIN[0], n - ORIGIN[1]
