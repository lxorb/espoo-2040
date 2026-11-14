import sys

from sim import City


def act(c, rank, pick="jam", first=True, maint=2):
    st, cat = c.state(), {p["id"]: p for p in c.catalogue()}
    y = st["step"]
    grow = {a["name"].split(" and ")[0]: (a["pop"] + a["jobs"]) / (a["pop0"] + a["jobs0"]) - 1 for a in st["areas"]}
    acc = [k for k in cat if k.endswith("-access")]
    if pick == "jam":
        acc.sort(key=lambda k: -cat[k]["jam"])
    else:
        acc = [k for n in sorted(grow, key=lambda n: -grow[n]) if grow[n] > 0.02 for k in acc if cat[k]["name"].startswith(n)]
    gen = [k for k in rank if k in cat]
    todo = (gen + acc[:1] if first else acc[:1] + gen) if y else gen
    out = []
    for k in todo:
        if len(out) < 2 and c.check(out + [{"build": k}])[-1] is None:
            out.append({"build": k})
        elif k.endswith("-access"):
            break
    return out + [{"maintain": maint}]


def run(seed, pick="jam", first=True, maint=2):
    c = City(seed, who="reference")
    base = sum(r["delay"] for r in c.forecast({}, 99))
    g = {k: base - sum(r["delay"] for r in c.forecast({0: [{"build": k}]}, 99)) for k in c.menu if not k.endswith("-access")}
    rank = sorted([k for k in g if g[k] > 0], key=lambda k: -g[k] / c.cat[k]["cost"])[:3]
    while not c.done:
        c.step(act(c, rank, pick, first, maint) if c.s["year"] < c.years else [])
    return c


if __name__ == "__main__":
    c = run(int(sys.argv[1]))
    print(c.seed, round(c.score(), 3), [(r["year"], [a["build"] for a in r["actions"] if a.get("ok") and "build" in a]) for r in c.recs if any(a.get("ok") and "build" in a for a in r["actions"])])
