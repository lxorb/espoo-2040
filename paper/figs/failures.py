import datetime
import hashlib
import itertools
import json
import platform
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, get_context
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from sim import City
from sim import params as P

EV, CACHE = ROOT / "evidence", ROOT / "runs" / "cf"
CAT = {p["id"]: p for p in json.loads((ROOT / "data" / "catalogue.json").read_text())}
ACC = {p["name"].split(" access")[0]: k for k, p in CAT.items() if k.endswith("-access")}
load = lambda p: json.loads(Path(p).read_text())
fold = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


class Rec(City):
    def _segs(self):
        self.seglog = super()._segs()
        return self.seglog


def acc(area):
    return ACC[area.split(" and ")[0]]


def truth(seed):
    c = City(seed)
    rate = {n: float((c.rp[m].mean() + c.rj[m].mean()) / 2) for n, m in c.areas}
    base = float(json.loads((ROOT / "data" / "out" / "zones.json").read_text())["growth"]["pop"])
    return dict(boom=sorted(n for n, r in rate.items() if r > base + 1e-12), fastest=max(rate, key=rate.get), rate=rate)


def files_key():
    return hashlib.sha1(b"".join(f.read_bytes() for f in sorted((ROOT / "sim").glob("*.py")) + [ROOT / "sim" / "norms.json"])).hexdigest()


def play(seed, plan, add=None, maint=None):
    key = hashlib.sha1(json.dumps([files_key(), Path(__file__).read_text(), seed, plan, add, maint]).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return load(f)
    c, years, started, added = Rec(seed), [], {}, None
    for y in range(c.years + c.runout):
        a = [dict(x) for x in (plan[y] if y < len(plan) else [])] if y < c.years else []
        if maint is not None and y < c.years:
            a = [x for x in a if "maintain" not in x] + [{"maintain": maint}]
        if add and added is None and y >= 2 and y < c.years and add not in started and c.check(a + [{"build": add}])[-1] is None:
            a, added = a + [{"build": add}], y
        cash, ok = float(c.s["cash"]), [k for k, m in zip(c.menu, c.mask()) if m]
        r = c.step(a)
        ids, inv, cond, cost = c.seglog
        first = sum(CAT[x["build"]]["cost"] / max(CAT[x["build"]]["years"], 1) for x in r["actions"] if x["ok"] and "build" in x)
        left0 = left = min(P.MAINT[r["maint"]], max(cash - first, 0.0))
        fix = np.zeros(ids.size, bool)
        for g in np.argsort(cond):
            if cond[g] < P.MAINT_BELOW and cost[g] <= left:
                fix[g], left = True, left - cost[g]
        rebuild = float(P.REBUILD * cost[(cond < P.COND_MIN) & ~fix].sum())
        for x in r["actions"]:
            if x["ok"] and "build" in x:
                started.setdefault(x["build"], y)
        years.append(dict(cash=cash, startable=ok, maint=r["maint"], first=first, maint_spend=float(left0 - left), rebuild=rebuild, spent=float(r["spent"]), delay=float(r["delay"]),
                          rejected=[x for x in r["actions"] if not x["ok"]]))
    res = c.result()
    out = dict(seed=seed, add=add, maint=maint, added=added, delay=res["delay"], raw=res["score"], score=min(1.0, max(0.0, res["score"])), started=started,
               rejected=sum(len(y["rejected"]) for y in years), split_off=max(abs(y["spent"] - y["first"] - y["maint_spend"] - y["rebuild"]) for y in years),
               cash_end=float(c.s["cash"]), years=years)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def ref_plan(seed):
    from handover import play as closed
    return [[{"build": b} for b in y["built"]] + [{"maintain": y["maint"]}] * (y["maint"] is not None) for y in closed(seed, [], 0, "agent")["years"]]


def runs():
    out = []
    for p in sorted(EV.glob("*/plan.json")) + sorted(EV.glob("*/*/plan.json")):
        d = load(p)
        if d.get("valid") is False:
            continue
        t, job = p.with_name("trajectory.json"), str(p.parent.relative_to(EV))
        out.append(dict(job=job, seed=int(d["seed"]), who=d["who"], role="reference" if d["who"] == "reference" else job.split("-s")[0],
                        plan=[[{k: v for k, v in a.items() if k != "ok"} for a in acts] for _, acts in d["plan"]][:15], stored=d.get("delay"),
                        steps=load(t)["steps"] if t.exists() else None))
    miss = sorted({r["seed"] for r in out} - {r["seed"] for r in out if r["role"] == "reference"})
    with ProcessPoolExecutor(max(len(miss), 1), get_context("spawn")) as p:
        out += [dict(job=f"policy-s{s}", seed=s, who="reference", role="reference", plan=pl, stored=None, steps=None) for s, pl in zip(miss, p.map(ref_plan, miss))]
    return out


def groups(rows):
    return sorted({r["role"] for r in rows} - {"reference"})


def ci(v, n=10000):
    v = np.asarray(v, float)
    if not v.size:
        return dict(mean=None, lo=None, hi=None, n=0)
    b = v[np.random.default_rng(0).integers(0, v.size, (n, v.size))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(v.size))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    sign = float(stats.binomtest(int((nz > 0).sum()), nz.size, 0.5, alternative="greater").pvalue) if nz.size else None
    wil = float(stats.wilcoxon(nz, alternative="greater").pvalue) if nz.size else None
    return dict(n=int(d.size), wins=int((d > 0).sum()), ties=int((d == 0).sum()), sign_p=sign, wilcoxon_p=wil)


def by_seed(rs, f):
    s = {}
    for r in rs:
        v = f(r)
        if v is not None:
            s.setdefault(r["seed"], []).append(v)
    return {k: float(np.mean(v)) for k, v in sorted(s.items())}


def meta(seeds, name):
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    return dict(script=f"paper/figs/{name}.py", command=f"uv run --with mcp==1.30.0 python paper/figs/{name}.py", seeds=sorted(seeds),
                date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), machine=f"{platform.node()} {cpu}", sim=files_key())


def year_of(steps):
    y = 0
    for s in steps:
        yield y, s
        y += sum(c.get("function_name") == "espoo_commit_year" for c in s.get("tool_calls") or [])


def mentions(steps, area):
    pat = re.compile(r"\b(" + re.escape(fold(area.split(" and ")[0])) + "|" + re.escape(acc(area)) + r")\b")
    msg = any_ = None
    for y, s in year_of(steps):
        if s.get("source") != "agent":
            continue
        if msg is None and pat.search(fold(s.get("message") or "")):
            msg = y
        if any_ is None and (y == msg or any(pat.search(fold(json.dumps(c.get("arguments"), ensure_ascii=False))) for c in s.get("tool_calls") or [])):
            any_ = y
    return msg, any_


def analyse(r, rep, tr):
    builds = set(rep["started"])
    boom = tr["boom"]
    yr = lambda y: None if y is None else P.Y0 + y
    see = []
    for a in boom:
        m, any_ = mentions(r["steps"], a) if r["steps"] else (None, None)
        act = rep["started"].get(acc(a))
        see.append(dict(area=a, project=acc(a), first_mention_msg=yr(m), first_mention=yr(any_), first_upgrade=yr(act),
                        lag=None if any_ is None or act is None else act - any_))
    up = lambda y: any(rep["started"].get(acc(a), 99) < y for a in boom)
    allup = lambda y: all(rep["started"].get(acc(a), 99) < y for a in boom)
    plan_years = rep["years"][:15]
    lock = sum(1 for y, v in enumerate(plan_years) if v["cash"] < 120 and not allup(y))
    lock_none = sum(1 for y, v in enumerate(plan_years) if v["cash"] < 120 and not up(y))
    todo = lambda y: [acc(a) for a in boom if rep["started"].get(acc(a), 99) >= y]
    free = sum(1 for y, v in enumerate(plan_years) if set(todo(y)) & set(v["startable"]) and not any(rep["started"].get(k) == y for k in todo(y)))
    blocked = sum(1 for y, v in enumerate(plan_years) if todo(y) and not set(todo(y)) & set(v["startable"]))
    total = sum(CAT[k]["cost"] for k in rep["started"]) + sum(v["maint_spend"] + v["rebuild"] for v in rep["years"])
    m2 = sum(v["maint_spend"] for v in plan_years if v["maint"] == 2)
    calls = [c for s in r["steps"] or [] for c in s.get("tool_calls") or []]
    fc = [c for c in calls if c.get("function_name") == "espoo_forecast"]
    fa = [sorted({x["build"] for v in (c.get("arguments") or {}).get("plan", {}).values() for x in v if isinstance(x, dict) and str(x.get("build", "")).endswith("-access")}) for c in fc]
    fa = [x for x in fa if x]
    hz = [c["arguments"]["years"] for c in fc if isinstance((c.get("arguments") or {}).get("years"), int)]
    by = sorted(P.Y0 + rep["started"][acc(a)] for a in boom if acc(a) in rep["started"])
    fst = min(((y, k) for k, y in rep["started"].items() if k.endswith("-access")), default=None)
    return dict(job=r["job"], seed=r["seed"], who=r["who"], role=r["role"], score=rep["score"], replay_off=None if r["stored"] is None else abs(rep["delay"] - r["stored"]) / r["stored"],
                builds=sorted(builds), boom=boom, fastest=tr["fastest"],
                sees_acts=see, boom_upgraded=sum(s["first_upgrade"] is not None for s in see), boom_upgraded_by_2031=sum((s["first_upgrade"] or 9999) <= 2031 for s in see),
                cash_lock_years=lock, cash_lock_years_no_upgrade=lock_none, boom_access_blocked_years=blocked, boom_access_skipped_years=free, maint2_years=sum(v["maint"] == 2 for v in plan_years),
                maint2_spend=m2, total_spend=total, maint2_share=m2 / total if total else None, maint_spend=sum(v["maint_spend"] for v in rep["years"]),
                rebuild_spend=sum(v["rebuild"] for v in rep["years"]), cash_end=rep["cash_end"], split_off=rep["split_off"],
                forecasts_with_access=len(fa) if r["steps"] else None, forecasts_access_never_started=sum(not set(x) & builds for x in fa) if r["steps"] else None,
                calls={k: sum(c.get("function_name") == f"espoo_{k}" for c in calls) for k in ("get_state", "list_projects", "forecast", "commit_year")} if r["steps"] else None,
                schedule=sorted([P.Y0 + y, k] for k, y in rep["started"].items()), maint=[v["maint"] for v in plan_years],
                first_access=None if fst is None else fst[1], first_access_year=None if fst is None else P.Y0 + fst[0],
                first_boom_year=by[0] if by else None, second_boom_year=by[1] if len(by) > 1 else None, two_boom=int(len(by) > 1),
                boom_in_window=sum(2028 <= y <= 2035 for y in by), forecast_horizon_mean=float(np.mean(hz)) if hz else None, forecast_horizon_max=max(hz) if hz else None,
                forecasts=len(fc) if r["steps"] else None)


def jaccard(rows):
    for r in rows:
        o = [len(set(r["builds"]) & set(x["builds"])) / max(len(set(r["builds"]) | set(x["builds"])), 1) for x in rows if x["role"] == r["role"] and x["seed"] != r["seed"]]
        r["jaccard_other_seeds"] = float(np.mean(o)) if o else None
    return {role: dict(pairs=ci([len(set(a["builds"]) & set(b["builds"])) / max(len(set(a["builds"]) | set(b["builds"])), 1)
                                for a, b in itertools.combinations([x for x in rows if x["role"] == role], 2) if a["seed"] != b["seed"]]))
            for role in groups(rows) + ["reference"]}


def job(r):
    return play(r["seed"], r["plan"])


def main():
    rs = runs()
    with Pool(min(len(rs), 16)) as p:
        reps = p.map(job, rs)
        trs = dict(zip(sorted({r["seed"] for r in rs}), p.map(truth, sorted({r["seed"] for r in rs}))))
    rows = [analyse(r, rep, trs[r["seed"]]) for r, rep in zip(rs, reps)]
    jac = jaccard(rows)
    keys = ["score", "jaccard_other_seeds", "boom_upgraded", "boom_upgraded_by_2031", "cash_lock_years", "cash_lock_years_no_upgrade", "boom_access_blocked_years", "boom_access_skipped_years", "maint2_years", "maint2_share", "maint2_spend", "maint_spend", "rebuild_spend", "total_spend", "cash_end",
            "forecasts_with_access", "forecasts_access_never_started", "two_boom", "boom_in_window", "first_access_year", "first_boom_year", "second_boom_year",
            "forecasts", "forecast_horizon_mean", "forecast_horizon_max"]
    summ = {"jaccard_pairs": jac}
    for role in groups(rows) + ["reference"]:
        rr = [r for r in rows if r["role"] == role]
        summ[role] = {k: dict(ci(list(by_seed(rr, lambda r: r[k]).values())), per_seed=by_seed(rr, lambda r: r[k])) for k in keys}
        summ[role]["lag"] = ci([s["lag"] for r in rr for s in r["sees_acts"] if s["lag"] is not None])
        summ[role]["never_upgraded_mentioned"] = sum(s["first_upgrade"] is None and s["first_mention"] is not None for r in rr for s in r["sees_acts"])
        summ[role]["boom_total"] = sum(len(r["boom"]) for r in rr)
        summ[role]["calls"] = {k: ci([r["calls"][k] for r in rr if r["calls"]]) for k in ("get_state", "list_projects", "forecast", "commit_year")}
    rf = by_seed([r for r in rows if r["role"] == "reference"], lambda r: r["score"])
    for g in groups(rows):
        ag = by_seed([r for r in rows if r["role"] == g], lambda r: r["score"])
        common = sorted(set(ag) & set(rf))
        summ[f"reference_vs_{g}_score"] = paired([rf[s] for s in common], [ag[s] for s in common])
        if g != "a1":
            base = by_seed([r for r in rows if r["role"] == "a1"], lambda r: r["score"])
            cm = sorted(set(ag) & set(base))
            summ[f"{g}_vs_a1_score"] = dict(paired([ag[s] for s in cm], [base[s] for s in cm]), seeds=cm, mean=float(np.mean([ag[s] for s in cm])) if cm else None,
                                            a1_mean=float(np.mean([base[s] for s in cm])) if cm else None, reference_mean=float(np.mean([rf[s] for s in cm])) if cm else None)
        for k in ("boom_access_blocked_years", "maint2_share", "jaccard_other_seeds"):
            a, b = (by_seed([r for r in rows if r["role"] == x], lambda r: r[k]) for x in (g, "reference"))
            summ[f"{g}_gt_reference_{k}"] = paired([a[s] for s in common], [b[s] for s in common])
    out = dict(meta=meta({r["seed"] for r in rs}, "failures"), runs=rows, summary=summ)
    (EV / "failures.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = main()
    for r in o["runs"]:
        print(r["job"], r["role"], round(r["score"], 3), "jac", r["jaccard_other_seeds"] and round(r["jaccard_other_seeds"], 2), "lock", r["cash_lock_years"], r["cash_lock_years_no_upgrade"], "blk", r["boom_access_blocked_years"], "skip", r["boom_access_skipped_years"], "spend", round(r["maint2_spend"]), round(r["maint_spend"]), round(r["rebuild_spend"]), round(r["total_spend"]),
              "m2", r["maint2_share"] and round(r["maint2_share"], 3), "fc", r["forecasts_with_access"], r["forecasts_access_never_started"], "calls", r["calls"],
              [(s["area"], s["first_mention"], s["first_upgrade"]) for s in r["sees_acts"]], "off", r["replay_off"], "split", round(r["split_off"], 6))
