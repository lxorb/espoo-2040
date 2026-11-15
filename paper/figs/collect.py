import datetime
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT = HERE / "data" / "runs.json"
RL, CUT = ROOT / "runs" / "rl", 5280
pick = lambda d, ks: {k: d[k] for k in ks if k in d}
load = lambda p: json.loads(Path(p).read_text())


def summary(d):
    y = d["years"]
    return dict(delay=[r["delay"] for r in y], spent=[r["spent"] for r in y],
                builds=[[a["build"] for a in r["actions"] if a.get("ok") and "build" in a] for r in y], maint=[r["maint"] for r in y])


def harbor(prefix):
    out = {}
    for job in sorted(glob.glob(str(ROOT / "jobs" / f"{prefix}-s*"))):
        log = glob.glob(f"{job}/*/verifier/run.json")
        if not log:
            continue
        d, st = load(log[0]), load(f"{job}/result.json")
        r = out[str(d["seed"])] = summary(d)
        r["usd"] = st["stats"].get("cost_usd")
        if st.get("finished_at"):
            r["minutes"] = (datetime.datetime.fromisoformat(st["finished_at"]) - datetime.datetime.fromisoformat(st["started_at"])).total_seconds() / 60
        for tr in glob.glob(f"{job}/*/agent/trajectory.json"):
            calls = [c.get("function_name", "") for x in load(tr)["steps"] for c in (x.get("tool_calls") or [])]
            r["calls"] = {k: calls.count(k) for k in set(calls)}
    return out


def main(*srcs):
    out = {k: v for k, v in (load(OUT) if OUT.exists() else {}).items() if "@" in k}
    for spec in srcs:
        tag, src = spec.split("=", 1)
        for f in sorted(glob.glob(f"{src}/rl-*.json")):
            name, seed = Path(f).stem[3:].rsplit("-", 1)
            out.setdefault(f"{name}@{tag}", {})[seed] = summary(load(f))
    out.update(agent=harbor("gate4"), refrl=harbor("ref3"), refold=harbor("refrl"), viz=dict(best=summary(load(ROOT / "viz" / "data" / "runs" / "best.json"))))
    OUT.write_text(json.dumps(out))
    st = load(RL / "strategy.json")
    curve = lambda n: [pick(r, ("it", "episodes", "mean_net")) for r in load(RL / "vm" / "6fae939a" / "runs" / "rl" / f"{n}-6fae939a.json") if "mean_net" in r and r.get("episodes", 0) <= CUT]
    keys = ("episodes", "score", "maintain", "access_share", "boom_hit", "entropy", "distinct", "freq", "first_access_year", "builds")
    (HERE / "data" / "rl.json").write_text(json.dumps(dict(
        strategy={k: [pick(r, keys) for r in v if r["episodes"] <= CUT] for k, v in st.items()}, curves={n: curve(n) for n in ("ppocold2", "gz2")},
        search=pick(load(RL / "search-eval.json"), ("seeds", "scores")), ablation=[pick(r, ("experts", "jam", "scores")) for r in load(RL / "ablation-jam.json")["rows"]],
        heldout=[pick(r, ("policy", "part", "seeds", "saved_frac", "mean")) for r in load(RL / "heldout.json")])))
    g = load(ROOT / "runs" / "grpo" / "curve.json")
    gkeys = ("step", "score", "sample_distinct", "maintain", "boom_hit", "boom_share", "train_std", "valid", "freq", "scores")
    (HERE / "data" / "grpo.json").write_text(json.dumps(dict(
        evals=[dict(pick(e, ("step", "score", "valid")), seeds=[dict(seed=x["seed"], greedy=pick(x.get("greedy") or {}, ("saving", "plan", "maintain"))) for x in e["seeds"]]) for e in g["evals"]],
        strategy=[pick(r, gkeys) for r in load(ROOT / "runs" / "grpo" / "strategy.json") if r.get("run", "A") == "A" and r["step"] >= 0])))


if __name__ == "__main__":
    main(*sys.argv[1:])
