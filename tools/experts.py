import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools")]
import norms as N

OUT = ROOT / "runs" / "expert"


def one(seed):
    path = OUT / f"{seed}.json"
    if path.exists():
        return seed
    hi, menu = N.play(seed, lambda c: [], income=1e6)[0], N.City(seed).menu
    val = {k: hi - N.play(seed, lambda c, k=k: [N.B(k)] if c.s["year"] == 0 else [], income=1e6)[0] for k in menu}
    o = [k for k in sorted(val, key=lambda k: -val[k]) if val[k] > 0]
    big = [k for k in o if val[k] > 3e6]
    loss, plan, mt = min((*N.play(seed, N.build(order), mt)[:2], mt) for order in (o, o[:8], o[:6], o[:4], o[:3], big + [k for k in o if k not in big]) for mt in (1, 2))
    c, ys = N.City(seed, who="expert"), []
    while not c.done:
        y = c.s["year"]
        acts = [N.B(k) for yy, k in plan if yy == y] + ([{"maintain": mt}] if y < c.years else [])
        ys.append(dict(actions=acts, delay=c.step(acts)["delay"]))
    tmp = path.with_suffix(f".{os.getpid()}")
    tmp.write_text(json.dumps(dict(seed=seed, loss=sum(y["delay"] for y in ys), plan=plan, maint=mt, years=ys)))
    os.replace(tmp, path)
    return seed


if __name__ == "__main__":
    lo, hi = map(int, sys.argv[1].split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(int(sys.argv[2])) as pool:
        for s in pool.imap_unordered(one, range(lo, hi + 1)):
            print(s, flush=True)
