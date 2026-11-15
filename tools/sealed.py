import json
import os
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from variants import PROBE, make


def run(task, agent, job, *extra, **env):
    subprocess.run(["harbor", "run", "-y", "-p", str(task), "-a", agent, "--job-name", job, *extra], cwd=ROOT, env={**os.environ, "SEED": "0", **env},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    trial = next((ROOT / "jobs" / job).glob("*/result.json")).parent
    r = json.loads((trial / "result.json").read_text())
    return trial, (r.get("verifier_result") or {}).get("rewards", {}).get("reward"), (r.get("exception_info") or {}).get("exception_type")


def main():
    tag = os.environ.get("TAG", platform.node())
    sealed, probe = make("sealed"), make("sealed-probe")
    out = {}
    for name, task, agent in (("oracle", sealed, "oracle"), ("nop", sealed, "nop"), ("probe", probe, "oracle")):
        trial, reward, err = run(task, agent, f"sealed-{tag}-{name}")
        out[name] = dict(reward=reward, error=err)
        if name == "probe":
            txt = next((trial / "agent").glob("*.txt")).read_text()
            out[name]["hosts"] = {u: c for u, c in re.findall(r"^probe (\S+) (\d+)", txt, re.M)}
    if os.environ.get("OPENCODE"):
        trial, reward, err = run(sealed, "opencode", f"sealed-{tag}-opencode", "-m", "openai/gpt-6.1-sol", OPENAI_API_KEY="sk-invalid-probe")
        log = "".join(p.read_text(errors="ignore") for p in (trial / "agent").rglob("*.txt"))
        out["opencode"] = dict(error=err, installed=(trial / "agent" / "setup").exists(), reached_api="invalid_api_key" in log)
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.machine())
    meta = dict(script="tools/sealed.py", command="OPENCODE=1 python3 tools/sealed.py", seeds=[0], date=str(date.today()),
                machine=f"{platform.node()} {cpu}", probe=PROBE)
    f = ROOT / "evidence" / "sealed.json"
    blob = json.loads(f.read_text()) if f.exists() else {}
    blob[tag] = dict(meta=meta, **out)
    f.write_text(json.dumps(blob, indent=1))
    print(json.dumps(blob[tag], indent=1))


if __name__ == "__main__":
    main()
