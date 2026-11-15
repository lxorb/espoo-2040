import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "task" / "tests")]
import exploits
from sim import City

FULL = bool(os.environ.get("ESPOO_FULL"))
SEEDS = list(range(16)) if FULL else [0, 1]
OWN = ["nothing", "maintain-0", "maintain-1", "maintain-2", "cheapest", "build-all", "illegal", "malformed", "blind-access-m0", "boom-access-m0"]
SRC = sorted((ROOT / "sim").glob("*")) + [ROOT / p for p in ("task/environment/server/server.py", "task/tests/test.sh", "task/tests/exploits.py", "rl/policy.py", "rl/policy.npz")]
CACHE = ROOT / "runs" / "pytest" / hashlib.sha1(b"".join(p.read_bytes() for p in SRC if p.is_file())).hexdigest()[:12]
plan = lambda job: [[{k: v for k, v in a.items() if k != "ok"} for a in acts] for _, acts in json.loads((ROOT / "evidence" / job / "plan.json").read_text())["plan"]]


def run(a):
    exploits.CACHE = CACHE
    return exploits.play(*a)


@pytest.fixture(scope="module")
def res():
    jobs = [(n, s) for n in ["reference"] + OWN for s in SEEDS]
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count()), get_context("spawn")) as p:
        return {(r["name"], r["seed"]): r for r in p.map(run, jobs)}


def replay(seed, p, years=15, runout=5):
    c = City(seed, years, runout)
    for a in p:
        if not c.done:
            c.step(a)
    while not c.done:
        c.step()
    return c


def test_deterministic():
    p = plan("ref3-s0")[:3]
    a, b = replay(0, p, 3, 0), replay(0, p, 3, 0)
    assert [r["delay"] for r in a.recs] == [r["delay"] for r in b.recs]
    assert all(np.array_equal(x["flow"], y["flow"]) and x["budget"] == y["budget"] for x, y in zip(a.recs, b.recs))


def test_verifier_matches_in_process():
    p = plan("ref3-s0")
    c = replay(0, p)
    state = dict(seed=0, years=15, runout=5, plan=p, seen=[r["delay"] for r in c.recs[:15]])
    r, out, _ = exploits.verify(state)
    assert r == min(1.0, max(0.0, c.score())) and json.loads(out.splitlines()[0])["replay_off"] == 0
    assert exploits.verify({**state, "seen": [0.0] * 15})[0] == r
    assert exploits.verify({**state, "plan": p[:3]})[0] == min(1.0, max(0.0, replay(0, p[:3]).score())) < r


def test_nothing_scores_zero(res):
    assert all(res["nothing", s]["reward"] < 1e-3 for s in SEEDS)


def test_bad_input_rejected(res):
    for n in ("illegal", "malformed"):
        assert all(not res[n, s]["accepted_bad"] and res[n, s]["reward"] < 1e-3 and not res[n, s]["after_done"] for s in SEEDS)


@pytest.mark.parametrize("name", OWN)
def test_below_reference_each_seed(res, name):
    assert all(res[name, s]["score"] < res["reference", s]["score"] for s in SEEDS)


@pytest.mark.skipif(not FULL, reason="ESPOO_FULL=1 runs seeds 0 to 15")
@pytest.mark.parametrize("name", OWN)
def test_mean_below_half(res, name):
    assert np.mean([res[name, s]["reward"] for s in SEEDS]) < 0.5
