import asyncio
import json
import os

import anyio
import numpy as np
from mcp.server.fastmcp import FastMCP

from sim import City
from sim import params as P

STATE = "/var/lib/env/state.json"
SEED, YEARS, RUNOUT = (int(os.environ.get(k) or d) for k, d in (("SEED", 0), ("YEARS", 15), ("RUNOUT", 5)))
TRIES, HORIZON = P.FORECASTS, P.FORECAST_YEARS
LEVELS = range(len(P.MAINT))
RULES = {"years": YEARS, "runout": RUNOUT, "boom_areas": list(P.BOOMS), "builds_per_year": P.BUILDS, "forecasts_per_year": TRIES, "forecast_years": HORIZON, "forecast_trust": f"step / (step + {P.LEARN:g})",
         "maintain": {"budget_meur": P.MAINT, "default": P.MAINT_DEFAULT, "runout_level": 0, "fixes_below_cond": P.MAINT_BELOW},
         "emergency_rebuild": {"below_cond": P.COND_MIN, "cost_x_resurface": P.REBUILD, "capacity": P.REBUILD_CAP}}

c = City(SEED, YEARS, RUNOUT)
road = {int(o): n for o, n in zip(c.s["osm"], (l["name"] for l in json.loads((c.data / "net.json").read_text())["links"])) if n}
lock = asyncio.Lock()
run = {"plan": [], "seen": [], "tries": TRIES, "last": c.forecast({}, 1)[0]}
mcp = FastMCP("espoo", host="0.0.0.0", port=8000, stateless_http=True, json_response=True)


def num(x):
    if isinstance(x, dict):
        return {k: num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [num(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 3) if abs(x) < 10 else round(float(x), 1) if abs(x) < 1000 else round(float(x))
    return x.item() if isinstance(x, np.generic) else x


def out(d):
    return json.dumps(num(d), separators=(",", ":"), ensure_ascii=False)


def save():
    with open(STATE + ".tmp", "w") as f:
        json.dump({"seed": SEED, "years": YEARS, "runout": RUNOUT, "plan": run["plan"], "seen": run["seen"]}, f)
    os.replace(STATE + ".tmp", STATE)


def clean(actions):
    if not isinstance(actions, list) or not all(isinstance(a, dict) and len(a) == 1 for a in actions):
        raise ValueError('actions must be a list of {"build": id} and at most one {"maintain": level}')
    b = [a["build"] for a in actions if "build" in a]
    m = [a["maintain"] for a in actions if "maintain" in a]
    if len(b) + len(m) < len(actions) or len(b) > P.BUILDS or len(m) > 1:
        raise ValueError(f"at most {P.BUILDS} builds and 1 maintain per year, no other keys")
    if any(not isinstance(i, str) or i not in c.menu for i in b):
        raise ValueError(f"unknown project in {b}")
    if any(type(k) is not int or k not in LEVELS for k in m):
        raise ValueError(f"maintain must be one of {list(LEVELS)}")
    return [dict(a) for a in actions]


def year(r):
    ev = r["events"]
    return {"year": r["year"], "loss_kh": r["delay"] / 1000, "jam_kh": r["jam"] / 1000, "car": r["car"], "cash_left": r["budget"],
            "rejected": [{k: v for k, v in a.items() if k != "ok"} for a in r["actions"] if not a["ok"]],
            "events": ev[:12] + [f"{len(ev) - 12} more"] * (len(ev) > 12)}


def roads(n=15):
    r = run["last"]
    flow, cap = np.asarray(r["flow"]), np.asarray(r["cap"]) * c.s["cap0"]
    late = flow * np.maximum(np.asarray(r["time"]) - c.s["len"] / c.s["kmh"] * 3.6, 0) / 3600 * P.ANNUAL / 1000
    rows = {}
    for o, f, v, d, k in zip(c.s["osm"], flow, flow / np.maximum(cap, 1), late, c.s["cond"]):
        name = road.get(int(o))
        if name:
            x = rows.setdefault(name, [name, 0.0, 0.0, 0.0, 1.0])
            x[1], x[2], x[3], x[4] = max(x[1], v), x[2] + d, max(x[3], f), min(x[4], k)
    top = sorted(rows.values(), key=lambda x: -x[2])[:n]
    return {"cols": ["road", "vc", "delay_kvh", "flow", "cond_min"], "rows": top}


def advance(a):
    f = c.fork(False)
    r = f.step(a)
    c.s, c.recs = f.s, f.recs
    return r


def status(k):
    return "built" if k in c.s["built"] else f"building {c.s['prog'][k]}" if k in c.s["prog"] else "open"


@mcp.tool(description=(
    "Current state: year, years and runout left, cash, income, opex, committed instalments, last year's loss and jam "
    "(thousand person-hours), car share, road condition (network mean, mean of the worst tenth), each development area's population and jobs now and in the "
    "base year (the growth signal), projects under construction and built, projects you can start now, forecasts left "
    "this year, the rules, and the 15 most delayed roads."
))
async def get_state():
    async with lock:
        st = c.state()
        return out({**{k: v for k, v in st.items() if k not in ("pop", "jobs")},
                    "years_left": max(YEARS - c.s["year"], 0), "runout_left": min(RUNOUT, YEARS + RUNOUT - c.s["year"]),
                    "startable": [k for k, ok in zip(c.menu, c.mask()) if ok], "forecasts_left": run["tries"],
                    "rules": RULES,
                    "roads": roads()})


@mcp.tool(description=(
    "The project menu: id, name, kind, cost (MEUR, equal instalments over the build years), build years, opex (MEUR per "
    "year once open), status (open, building N = years left, built), jam (share of last year's network delay on the roads it "
    "changes, 0 once started) and the existing roads it changes."
))
async def list_projects():
    async with lock:
        rows, jam = [], {p["id"]: p["jam"] for p in c.catalogue()}
        for k in c.menu:
            p = c.cat[k]
            osm = {o for e in p.get("mod", []) + p.get("works", []) + [p.get("toll") or {"osm": []}] for o in e["osm"]}
            rows.append([k, p["name"], p["kind"], p["cost"], p["years"], p.get("opex", 0), status(k), jam.get(k, 0), sorted({road[o] for o in osm if o in road})[:6]])
        return out({"cols": ["id", "name", "kind", "cost", "years", "opex", "status", "jam", "roads"], "rows": rows})


@mcp.tool(description=(
    "Forecast a plan without changing the city. plan maps a year offset (\"0\" = this year) to that year's actions; years is "
    "how many years to simulate (at most rules.forecast_years and what is left, runout included). Growth blends each area's "
    "prior middle with the rate it has shown so far, weighted by rules.forecast_trust. Returns per year the loss and jam (thousand person-hours), car share, cash left and rejected actions. "
    "Limited forecasts per year; the count resets on commit_year."
))
async def forecast(plan: dict[str, list], years: int):
    left = min(HORIZON, YEARS + RUNOUT - c.s["year"])
    if type(years) is not int or not 1 <= years <= left:
        raise ValueError(f"years must be an integer from 1 to {left}")
    p = {}
    for k, v in plan.items():
        if not str(k).isdigit() or int(k) >= years:
            raise ValueError(f"plan key {k!r} must be a year offset from 0 to {years - 1}")
        p[int(k)] = clean(v)
    async with lock:
        if run["tries"] < 1:
            raise ValueError("no forecasts left this year")
        run["tries"] -= 1
        ys = [year(r) for r in await anyio.to_thread.run_sync(c.forecast, p, years)]
        return out({"years": [{k: v for k, v in y.items() if k != "events"} for y in ys], "total_loss_kh": sum(y["loss_kh"] for y in ys),
                    "feasible": not any(y["rejected"] for y in ys), "forecasts_left": run["tries"]})


@mcp.tool(description=(
    "Carry out this year's actions and advance one year. actions: up to rules.builds_per_year {\"build\": id} and one "
    "{\"maintain\": level} (index into rules.maintain.budget_meur, default rules.maintain.default). Rejected actions cost nothing. Returns the year's loss, jam, car share, cash left, rejected actions and events."
))
async def commit_year(actions: list):
    a = clean(actions)
    async with lock:
        if c.done:
            raise ValueError("the run is over")
        r = await anyio.to_thread.run_sync(advance, a)
        run.update(plan=run["plan"] + [a], seen=run["seen"] + [r["delay"]], tries=TRIES, last=r)
        save()
        return out({**year(r), "cash_next_year": c.s["cash"], "years_left": max(YEARS - c.s["year"], 0), "done": c.done})


if __name__ == "__main__":
    save()
    mcp.run(transport="streamable-http")
