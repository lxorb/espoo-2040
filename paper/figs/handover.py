import asyncio
import datetime
import hashlib
import json
import os
import platform
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "task" / "tests")]
EV, CACHE, KS = ROOT / "evidence", ROOT / "runs" / "handover", range(16)
SRC = sorted((ROOT / "sim").glob("*.py")) + [ROOT / p for p in ("sim/norms.json", "task/environment/server/server.py", "rl/policy.py", "rl/policy.npz", "paper/figs/handover.py")]
KEY = hashlib.sha1(b"".join(p.read_bytes() for p in SRC)).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
strip = lambda acts: [{k: v for k, v in a.items() if k not in ("ok", "why")} for a in acts]
S = POL = None


def runs():
    out = []
    for p in sorted(EV.glob("*/plan.json")) + sorted(EV.glob("*/*/plan.json")):
        d = load(p)
        if d["who"] != "reference" and d.get("valid") is not False:
            out.append(dict(job=str(p.parent.relative_to(EV)), seed=int(d["seed"]), plan=[strip(a) for _, a in d["plan"]][:15]))
    return out


def play(seed, plan, k, first):
    global S, POL
    f = CACHE / f"{hashlib.sha1(json.dumps([KEY, seed, plan, k, first]).encode()).hexdigest()}.json"
    if f.exists():
        return load(f)
    if S is None:
        import exploits
        S, POL = exploits.boot()
        S.STATE = f"{tempfile.mkdtemp()}/state.json"
    from sim import City
    S.SEED = seed
    S.c = City(seed, S.YEARS, S.RUNOUT)
    S.run = {"plan": [], "seen": [], "tries": S.TRIES, "last": S.c.forecast({}, 1)[0]}
    t = lambda tool, **a: json.loads(asyncio.run(S.mcp.call_tool(tool, a))[0].text)
    log = []
    for y in range(S.YEARS):
        mine = (y < k) == (first == "agent")
        if mine:
            acts = plan[y] if y < len(plan) else []
        else:
            st, rows = t("get_state"), t("list_projects")["rows"]
            proj = [dict(id=p[0], kind=p[2], cost=p[3], years=p[4], opex=p[5], jam=p[7]) for p in rows]
            acts = POL(st, proj, [p["id"] in st["startable"] for p in proj], years=st["rules"]["years"])
        r = t("commit_year", actions=acts)
        log.append(dict(year=2026 + y, by="agent" if mine else "reference", built=[a["build"] for a in acts if "build" in a and not any(x.get("build") == a["build"] for x in r["rejected"])],
                        maint=next((a["maintain"] for a in acts if "maintain" in a), None), rejected=r["rejected"]))
    while not S.c.done:
        S.c.step()
    res = S.c.result()
    out = dict(seed=seed, k=k, first=first, score=float(np.clip(res["score"], 0, 1)), raw=res["score"], delay=res["delay"], cash_end=float(S.c.s["cash"]), years=log)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def job(a):
    return play(*a)


def plot(rows):
    import make
    plt = make.plt
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.1), sharey=True)
    cm = plt.get_cmap("viridis")
    for r in rows:
        c, ls = cm(r["seed"] / 5), "-" if r["job"].startswith("gate4") else "--"
        for i, d in enumerate(("agent", "reference")):
            ax[i].plot(KS, r[d], ls, color=c, lw=0.9, label=f"seed {r['seed']}" if ls == "-" and i else None)
            if r[f"break_{d}"] is not None:
                ax[i].plot(r[f"break_{d}"], r[d][r[f"break_{d}"]], "o", color=c, ms=3)
    ax[0].set(xlabel="$k$ (agent, then learned policy)", ylabel="score", title="(a) agent first", xticks=range(0, 16, 3))
    ax[1].set(xlabel="$k$ (learned policy, then agent)", title="(b) learned policy first", xticks=range(0, 16, 3))
    ax[1].plot([], [], "-", color=make.MUTED, lw=0.9, label="run 1")
    ax[1].plot([], [], "--", color=make.MUTED, lw=0.9, label="run 2")
    ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.8)
    make.save(fig, "handover")


def main():
    rs = runs()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", os.cpu_count())), get_context("spawn")) as p:
        res = list(p.map(job, [(r["seed"], r["plan"], k, d) for r in rs for d in ("agent", "reference") for k in KS]))
    it, rows = iter(res), []
    for r in rs:
        a, b = [next(it) for _ in KS], [next(it) for _ in KS]
        ref, ag = a[0]["score"], a[15]["score"]
        mid = (ref + ag) / 2
        fa = next((k for k in KS if a[k]["score"] < mid), None)
        fb = next((k for k in KS if b[k]["score"] >= mid), None)
        rows.append(dict(job=r["job"], seed=r["seed"], reference_alone=ref, agent_alone=ag, gap=ref - ag, mid=mid, agent=[x["score"] for x in a], reference=[x["score"] for x in b],
                         break_agent=fa, break_reference=fb, break_year=None if fa is None else 2026 + fa - 1,
                         decision=None if not fa else dict(agent=a[fa]["years"][fa - 1], reference=a[fa - 1]["years"][fa - 1])))
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    meta = dict(script="paper/figs/handover.py", command="uv run --with mcp==1.30.0 python paper/figs/handover.py", seeds=sorted({r["seed"] for r in rs}),
                date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), machine=f"{platform.node()} {cpu}", sim=KEY,
                threshold="midpoint of the score of the reference alone (k=0) and the agent alone (k=15)")
    (EV / "handover.json").write_text(json.dumps(dict(meta=meta, runs=rows), indent=1))
    plot(rows)
    for r in rows:
        print(r["job"], r["seed"], "gap", round(r["gap"], 3), "break", r["break_agent"], r["break_year"], "rev", r["break_reference"],
              [round(x, 2) for x in r["agent"]], [round(x, 2) for x in r["reference"]], r["decision"])


if __name__ == "__main__":
    main()
