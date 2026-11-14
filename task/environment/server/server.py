import asyncio
import json
import os

import anyio
import numpy as np
from mcp.server.fastmcp import FastMCP

from sim import City
from sim import params as P

STATE = "/var/lib/env/state.json"
SEED, YEARS, RUNOUT, TRIES = (int(os.environ.get(k) or d) for k, d in (("SEED", 0), ("YEARS", 15), ("RUNOUT", 5), ("FORECASTS", 6)))
SORTS = {"vc": -1, "delay": -1, "flow": -1, "cond": 1, "id": 1}
COLS = ["id", "name", "cls", "lane_km", "flow", "vc", "delay_kvh", "cond", "resurface"]

c = City(SEED, YEARS, RUNOUT)
s = c.s
zj = json.loads((c.data / "zones.json").read_text())
ix = {z["id"]: i for i, z in enumerate(zj["zones"])}
areas = [(a["name"], [ix[z] for z in a["zones"]]) for a in zj["growth"]["areas"]]
road = {int(o): n for o, n in zip(s["osm"], (l["name"] for l in json.loads((c.data / "net.json").read_text())["links"])) if n}
lock = asyncio.Lock()
run = {"plan": [], "tries": TRIES, "last": c.forecast({}, 1)[0]}
mcp = FastMCP("espoo", host="0.0.0.0", port=8000, stateless_http=True, json_response=True)


def out(d):
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def save():
    with open(STATE + ".tmp", "w") as f:
        json.dump({"seed": SEED, "years": YEARS, "runout": RUNOUT, "plan": run["plan"]}, f)
    os.replace(STATE + ".tmp", STATE)


def segs():
    r = run["last"]
    ids, inv = np.unique(s["seg"], return_inverse=True)
    flow = np.asarray(r["flow"])
    vc = flow / np.maximum(np.asarray(r["cap"]) * s["cap0"], 1)
    late = np.maximum(np.asarray(r["time"]) - s["len"] / s["kmh"] * 3.6, 0)
    agg = lambda f, v, x: (f.at(o := np.full(ids.size, x), inv, v), o)[1]
    lk = np.bincount(inv, s["len"] * s["lanes"]) / 1000
    cls = np.empty(ids.size, object)
    cls[inv] = s["cls"]
    t = dict(id=ids, lane_km=lk.round(1), flow=agg(np.maximum, flow, 0.0).round(), vc=agg(np.maximum, vc, 0.0).round(2),
             delay=(np.bincount(inv, flow * late) / 3600 * P.ANNUAL / 1000).round(1), cond=agg(np.minimum, s["cond"], 1.0).round(2),
             resurface=(lk * P.RESURF).round(2))
    rows = {int(g): [int(g), s["segname"].get(int(g), ""), cls[k], float(t["lane_km"][k]), int(t["flow"][k]), float(t["vc"][k]),
                     float(t["delay"][k]), float(t["cond"][k]), float(t["resurface"][k])] for k, g in enumerate(ids)}
    return rows, t


def pick(sort, n, only=None):
    rows, t = segs()
    o = np.argsort(SORTS[sort] * t[sort], kind="stable")
    sel = [rows[int(t["id"][k])] for k in o if only is None or int(t["id"][k]) in only]
    return {"cols": COLS, "rows": sel[:n]}


def seg_id(v):
    if type(v) is not int or not (s["seg"] == v).any():
        raise ValueError(f"unknown seg {v!r}")
    return v


def clean(actions):
    if not isinstance(actions, list) or len(actions) > 1000:
        raise ValueError("actions must be a list of at most 1000 objects")
    for a in actions:
        k = set(a) if isinstance(a, dict) else None
        if k in ({"build"}, {"build", "level"}):
            if not isinstance(a["build"], str) or a["build"] not in c.cat:
                raise ValueError(f"unknown project {a['build']!r}")
            if type(a.get("level", 0)) is not int:
                raise ValueError("level must be an integer")
        elif k in ({"resurface"}, {"retime"}):
            seg_id(a[k.pop()])
        else:
            raise ValueError(f"bad action {a!r}: use {{build, level?}}, {{resurface}} or {{retime}}")
    return [dict(a) for a in actions]


def summary(r):
    ev = r["events"]
    return {"year": r["year"], "delay_kh": round(r["delay"] / 1000), "car": round(r["car"], 3), "cash_left": round(r["budget"], 1),
            "rejected": [{k: v for k, v in a.items() if k != "ok"} for a in r["actions"] if not a["ok"]],
            "events": ev[:20] + [f"{len(ev) - 20} more"] * (len(ev) > 20)}


@mcp.tool(description=(
    "Current state: year, cash available now, income, yearly opex, committed future instalments, last year's total delay "
    "(thousand person-hours), car share, population and jobs (total and growth so far per development area), projects under "
    "construction (years left) and built (level), forecasts left this year, 25 most congested and 25 worst-condition segments. "
    "Traffic figures are from the last committed year, or this year's do-nothing forecast before the first commit."
))
async def get_state():
    async with lock:
        st, r = c.state(), run["last"]
        pop, jobs = np.array(st["pop"]), np.array(st["jobs"])
        return out({
            "year": st["year"], "years_left": max(YEARS - s["year"], 0), "runout_left": min(RUNOUT, YEARS + RUNOUT - s["year"]),
            "done": c.done, "cash": round(st["budget"], 1), "income": st["income"], "opex": round(st["opex"], 2),
            "committed": {str(P.Y0 + y): round(v, 1) for y, v in sorted(s["due"].items())},
            "traffic_year": r["year"], "delay_kh": round(r["delay"] / 1000), "car": round(r["car"], 3),
            "pop": round(pop.sum()), "jobs": round(jobs.sum()),
            "areas": [[n, round(pop[m].sum()), round(100 * (pop[m].sum() / c.pop0[m].sum() - 1), 1), round(jobs[m].sum()),
                       round(100 * (jobs[m].sum() / max(c.jobs0[m].sum(), 1) - 1), 1)] for n, m in areas],
            "area_cols": ["name", "pop", "pop_growth_pct", "jobs", "jobs_growth_pct"],
            "building": st["works"], "built": st["built"], "forecasts_left": run["tries"],
            "congested": pick("vc", 25), "worst": pick("cond", 25),
        })


@mcp.tool(description=(
    "All catalogue projects: id, name, kind, cost (MEUR, paid in equal instalments over the build years), build years, "
    "opex (MEUR per year once open), levels (EUR price per level for toll and parking, pick one with level), status "
    "(open, building N = years left, built), affordable now, and the names of existing roads it changes."
))
async def list_projects():
    async with lock:
        rows = []
        for k, p in c.cat.items():
            st = "built" if k in s["built"] else f"building {s['prog'][k][0]}" if k in s["prog"] else "open"
            osm = {o for e in p.get("mod", []) + p.get("works", []) + [p.get("toll") or {"osm": []}] for o in e["osm"]}
            rows.append([k, p["name"], p["kind"], p["cost"], p["years"], p.get("opex", 0), (p.get("toll") or p.get("park") or {}).get("eur"),
                         st, st == "open" and c._afford(p["cost"], max(p["years"], 1)), sorted({road[o] for o in osm if o in road})[:6]])
        return out({"cols": ["id", "name", "kind", "cost", "years", "opex", "levels", "status", "afford", "roads"], "rows": rows})


@mcp.tool(description=(
    "Segment table from the last committed year (or this year's do-nothing forecast): id, name, class, lane-km, peak-hour "
    "flow (veh/h, busiest link), v/c (busiest link), delay (thousand vehicle-hours per year), condition (0 to 1, worst link), "
    "resurface cost (MEUR). Retiming costs 0.02 MEUR per segment. sort: vc, delay, flow (descending), cond, id (ascending). "
    "limit: rows to return (up to all segments). ids: optional list of segment ids to restrict to."
))
async def get_segments(sort: str = "vc", limit: int = 25, ids: list[int] | None = None):
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {list(SORTS)}")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    only = None if ids is None else {seg_id(v) for v in ids}
    async with lock:
        return out({**pick(sort, limit, only), "retime_cost": P.RETIME_COST})


@mcp.tool(description=(
    "Forecast a plan without changing the city. plan maps a year offset (\"0\" = this year) to a list of actions; years is "
    "how many years to simulate (at most what is left, runout included). Uses the expected growth of every area, not the "
    "actual one. Returns per year the total delay (thousand person-hours), car share, cash left after that year's spending, "
    "rejected actions with reasons, and whether the plan is feasible. Limited number of forecasts per year; resets on commit_year."
))
async def forecast(plan: dict[str, list], years: int):
    left = YEARS + RUNOUT - s["year"]
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
        recs = await anyio.to_thread.run_sync(c.forecast, p, years)
        ys = [summary(r) for r in recs]
        return out({"years": [{k: v for k, v in y.items() if k != "events"} for y in ys],
                    "total_delay_kh": round(sum(r["delay"] for r in recs) / 1000),
                    "feasible": not any(y["rejected"] for y in ys), "forecasts_left": run["tries"]})


@mcp.tool(description=(
    "Carry out this year's actions and advance one year. actions: list of {\"build\": id} or {\"build\": id, \"level\": k}, "
    "{\"resurface\": seg_id}, {\"retime\": seg_id}. Actions that do not fit the budget are rejected and cost nothing. "
    "Returns the year's delay, car share, cash left, rejected actions and events."
))
async def commit_year(actions: list):
    a = clean(actions)
    async with lock:
        if c.done:
            raise ValueError("the run is over")
        r = await anyio.to_thread.run_sync(c.step, a)
        run.update(plan=run["plan"] + [a], tries=TRIES, last=r)
        save()
        return out({**summary(r), "cash_next_year": round(s["cash"], 1), "years_left": max(YEARS - s["year"], 0), "done": c.done})


if __name__ == "__main__":
    save()
    mcp.run(transport="streamable-http")
