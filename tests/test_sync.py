import filecmp
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ["data/out/net.json", "data/out/zones.json", "data/out/od.npy", "data/out/truck.npy", "data/out/pt.npy", "data/catalogue.json"]
SIM = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "sim").glob("*.py"))] + ["sim/norms.json"]
PAIRS = [(f"{d}/{f}", f) for d in ("task/environment/server", "task/tests") for f in SIM + DATA]
PAIRS += [(f"task/solution/{f}", f"rl/{f}") for f in ("policy.py", "policy.npz", "places.json")]


@pytest.mark.parametrize("copy,src", PAIRS)
def test_copy(copy, src):
    assert filecmp.cmp(ROOT / copy, ROOT / src, shallow=False), f"{copy} differs from {src}: run ./eval.sh sync"


@pytest.mark.parametrize("d", ["task/environment/server/sim", "task/tests/sim"])
def test_no_extra(d):
    assert {p.name for p in (ROOT / d).iterdir() if p.name != "__pycache__"} == {Path(f).name for f in SIM}
