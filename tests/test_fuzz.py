import asyncio
import copy
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "task" / "tests")]
import exploits
from mcp.server.fastmcp.exceptions import ToolError

S, _ = exploits.boot()
S.STATE = str(Path(tempfile.mkdtemp()) / "state.json")
N = int(os.environ.get("ESPOO_FUZZ", 25))
CFG = settings(max_examples=N, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))

ids = st.sampled_from(S.c.menu + ["nope", "", "OLARI-LINK", "olari-link "])
val = st.one_of(st.none(), st.booleans(), st.integers(-3, 4), st.floats(allow_nan=True), st.text(max_size=4), ids, st.lists(ids, max_size=2))
anyjson = st.recursive(st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=4),
                       lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=4), c, max_size=3), max_leaves=8)
act = st.one_of(st.fixed_dictionaries({"build": ids}), st.fixed_dictionaries({"maintain": st.integers(-1, 3)}), st.fixed_dictionaries({"build": val}),
                st.fixed_dictionaries({"maintain": val}), st.dictionaries(st.sampled_from(["build", "maintain", "x"]), val, max_size=3))
near = st.lists(st.fixed_dictionaries({"build": ids}), max_size=2) | st.lists(st.fixed_dictionaries({"build": ids}), max_size=2).map(lambda l: l + [{"maintain": 2}])
actions = st.one_of(near, st.lists(act, max_size=4), anyjson)
plans = st.one_of(st.dictionaries(st.sampled_from(["0", "1", "2"]), near, max_size=3), anyjson, st.dictionaries(st.sampled_from(["0", "1", "2", "3", "-1", "00", "٣", "a"]) | st.text(max_size=2), actions, max_size=3))


def snap(tries=True):
    s = S.c.s
    return pickle.dumps((s["year"], s["cash"], sorted(s["due"].items()), sorted(s["prog"].items()), sorted(s["built"].items()), s["cond"].tobytes(), s["fp"].tobytes(),
                         len(S.c.recs), json.dumps(S.run["plan"]), list(S.run["seen"]), S.run["tries"] if tries else None))


def call(tool, **a):
    keep = copy.deepcopy(S.c.s), list(S.c.recs), dict(S.run)
    try:
        return json.loads(asyncio.run(S.mcp.call_tool(tool, a))[0].text), keep
    except ToolError:
        return None, keep


def restore(keep):
    S.c.s, S.c.recs, S.run = copy.deepcopy(keep[0]), list(keep[1]), dict(keep[2])


@CFG
@given(actions)
def test_commit_year_fuzz(a):
    before = snap()
    r, keep = call("commit_year", actions=a)
    if r is None:
        assert snap() == before
        return
    f = S.c.fork(False)
    f.s, f.recs = copy.deepcopy(keep[0]), list(keep[1])
    g = f.step([{k: v for k, v in y.items() if k not in ("ok", "why")} for y in S.c.recs[-1]["actions"] if y["ok"]])
    assert S.c.s["year"] == keep[0]["year"] + 1 and g["delay"] == S.c.recs[-1]["delay"] and f.s["cash"] == S.c.s["cash"]
    restore(keep)
    assert snap() == before


@CFG
@given(plans, st.one_of(st.integers(-1, 4), st.integers(1, 3), val))
def test_forecast_fuzz(p, years):
    before, tries = snap(False), S.run["tries"]
    r, keep = call("forecast", plan=p, years=years)
    assert snap(False) == before and S.run["tries"] == tries - (r is not None)
    restore(keep)


@CFG
@given(st.dictionaries(st.sampled_from(["0", "1", "2"]), near, max_size=3))
def test_forecast_near_valid(p):
    test_forecast_fuzz.hypothesis.inner_test(p, 3)
