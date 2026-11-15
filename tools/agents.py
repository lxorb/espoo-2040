import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}|api[_-]?key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_-]{16,}", re.I)


def seeds(s):
    return [x for p in s.split(",") for x in (range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1))]


def dirs(job):
    return [d for d in (ROOT / "jobs").glob(f"{job}-*") if d.name.rsplit("-", 1)[0] == job]


def done(job):
    return [r for d in dirs(job) for r in d.glob("*/result.json") if json.loads(r.read_text()).get("verifier_result")]


def run(a, seed):
    job = f"{a.job}-s{seed}"
    k = a.runs - len(done(job))
    if k > 0:
        n = len(dirs(job))
        cmd = ["harbor", "run", "-y", "-p", a.task.format(seed=seed), "--job-name", f"{job}-{n}", "-a", a.agent, "-k", str(k), "-n", str(min(k, a.concurrent))]
        cmd += ["-m", a.model] * bool(a.model) + [x for kv in a.ak for x in ("--ak", kv)]
        subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, SEED=str(seed)), check=False)


def clean(s):
    s = s.replace(f"file://{ROOT}/", "").replace(f"{ROOT}/", "").replace(str(Path.home()), "~")
    hits = SECRET.findall(s)
    if hits:
        print(f"redacted {len(hits)} secret-like strings", file=sys.stderr)
    return SECRET.sub("[redacted]", s)


def collect(job):
    sys.path.insert(0, str(ROOT))
    from sim import City
    for r in done(job):
        t = r.parent
        out = ROOT / "evidence" / job / f"{t.parent.name.rsplit('-', 1)[1]}-{t.name}"
        if (out / "plan.json").exists():
            continue
        st = json.loads((t / "artifacts" / "var" / "lib" / "env" / "state.json").read_text())
        res = json.loads(r.read_text())
        c = City(st["seed"], st["years"], st["runout"])
        for p in st["plan"]:
            c.step(p)
        plan = [[x["year"], x["actions"]] for x in c.recs]
        while not c.done:
            c.step()
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(dict(seed=st["seed"], who=res["config"]["agent"]["model_name"] or res["config"]["agent"]["name"],
                                                       delay=sum(x["delay"] for x in c.recs), score=c.score(),
                                                       harbor_reward=res["verifier_result"]["rewards"]["reward"], plan=plan)))
        if (t / "agent" / "trajectory.json").exists():
            (out / "trajectory.json").write_text(clean((t / "agent" / "trajectory.json").read_text()))
        (out / "result.json").write_text(clean(json.dumps(res, indent=1)))
        print(out, c.score())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "collect"])
    ap.add_argument("--job", required=True)
    ap.add_argument("--task", default="task")
    ap.add_argument("--agent", default="opencode")
    ap.add_argument("--model", default="")
    ap.add_argument("--seeds", default="16-47")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--ak", action="append", default=[])
    a = ap.parse_args()
    ss = seeds(a.seeds)
    if a.cmd == "run":
        with ThreadPoolExecutor(max(1, a.concurrent // a.runs)) as pool:
            list(pool.map(lambda s: run(a, s), ss))
    for s in ss:
        collect(f"{a.job}-s{s}")


if __name__ == "__main__":
    main()
