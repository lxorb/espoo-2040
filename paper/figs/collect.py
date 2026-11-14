import datetime
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
JOBS = dict(agent=os.environ.get("GATE", "gate4"), refrl="ref3", refold="refrl")


def summary(d):
    y = d["years"]
    return dict(delay=[r["delay"] for r in y], spent=[r["spent"] for r in y],
                builds=[[a["build"] for a in r["actions"] if a.get("ok") and "build" in a] for r in y], maint=[r["maint"] for r in y])


def harbor(name, prefix, out):
    for job in sorted(glob.glob(str(ROOT / "jobs" / f"{prefix}-s*"))):
        runs = glob.glob(f"{job}/*/verifier/run.json")
        if not runs:
            continue
        d = json.loads(Path(runs[0]).read_text())
        r = out.setdefault(name, {})[str(d["seed"])] = summary(d)
        st = json.loads(Path(job, "result.json").read_text())
        r["usd"] = st["stats"].get("cost_usd")
        if st.get("finished_at"):
            r["minutes"] = (datetime.datetime.fromisoformat(st["finished_at"]) - datetime.datetime.fromisoformat(st["started_at"])).total_seconds() / 60
        tr = glob.glob(f"{job}/*/agent/trajectory.json")
        if tr:
            calls = [c.get("function_name", "") for x in json.loads(Path(tr[0]).read_text())["steps"] for c in (x.get("tool_calls") or [])]
            r["calls"] = {k: calls.count(k) for k in set(calls)}


def main(*srcs):
    old = HERE / "data" / "runs.json"
    out = {k: v for k, v in (json.loads(old.read_text()) if old.exists() else {}).items() if "@" in k}
    for spec in srcs:
        tag, src = spec.split("=", 1)
        for f in sorted(glob.glob(f"{src}/rl-*.json")):
            name, seed = Path(f).stem[3:].rsplit("-", 1)
            out.setdefault(f"{name}@{tag}", {})[seed] = summary(json.loads(Path(f).read_text()))
    for name, prefix in JOBS.items():
        harbor(name, prefix, out)
    for n in ("best", "nothing"):
        f = ROOT / "viz" / "data" / "runs" / f"{n}.json"
        if f.exists():
            d = json.loads(f.read_text())
            out.setdefault("viz", {})[n] = dict(seed=d["seed"], **summary(d))
    (HERE / "data" / "runs.json").write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
