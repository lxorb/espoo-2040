import glob
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
CONST = ["otaniemi-access", "kiviruukki-access", "espoonvayla-2x2"]


def acts(c):
    return [[{k: v for k, v in a.items() if k in ("build", "maintain")} for a in r["actions"] if a.get("ok")] for r in c.recs[:c.years]]


def grpo(seed):
    from grpo import env
    i = env.info(seed)
    top = [env.access(n) for n, _, _ in i["areas"][:3]]
    lists = dict(const=(CONST, 2), growth_m1=(env.FIXED[:2] + top, 1), growth_m2=(env.FIXED[:2] + top, 2), growfirst_m1=(top + env.FIXED[:2], 1),
                 growfirst_m2=(top + env.FIXED[:2], 2), reflike_m1=(["espoonvayla-2x2"] + top[:2], 1), reflike_m2=(["espoonvayla-2x2"] + top[:2], 2))
    res = {k: env.play(seed, *v) for k, v in lists.items()}
    best = env.search(seed, max(res.values(), key=lambda r: r["frac"])["plan"], 40)
    out = [dict(src="grpo constant plan" if k == "const" else f"review grpo_headroom {k}", years=acts(env.execute(seed, *v))) for k, v in lists.items()]
    return seed, out + [dict(src="review grpo_headroom search", years=acts(env.execute(seed, best["plan"], best["maintain"])))]


def plans():
    out = {}
    for f in sorted(glob.glob(str(ROOT / "evidence/**/plan.json"), recursive=True)):
        d = json.loads(Path(f).read_text())
        if d.get("valid") is not False and 0 <= d["seed"] < 16:
            out.setdefault(str(d["seed"]), []).append(dict(src=f"evidence {Path(f).relative_to(ROOT / 'evidence').parts[0]} ({d.get('who')})",
                                                            years=[[{k: v for k, v in a.items() if k in ("build", "maintain")} for a in y[1]] for y in d["plan"]]))
    return out


if __name__ == "__main__":
    ex = plans()
    with mp.get_context("spawn").Pool(16) as pool:
        for s, e in pool.imap_unordered(grpo, range(16)):
            ex.setdefault(str(s), []).extend(e)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / sys.argv[1]).write_text(json.dumps(ex))
    print({s: len(v) for s, v in ex.items()})
