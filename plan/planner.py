import sys

from sim import City


def act(c, rank, spill=True, maint=2, cheap=15):
    st, cat = c.state(), {p["id"]: p for p in c.catalogue()}
    y = st["step"]
    grow = {a["name"].split(" and ")[0]: (a["pop"] + a["jobs"]) / (a["pop0"] + a["jobs0"]) - 1 for a in st["areas"]}
    acc = [k for k in cat if k.endswith("-access")]
    boom = {k for n in grow if y > 1 and grow[n] > 0.015 * (y - 1) for k in acc if cat[k]["name"].startswith(n)}
    if spill and boom:
        boom.add(max(acc, key=lambda k: cat[k]["jam"]))
    acc = sorted(boom, key=lambda k: -cat[k]["jam"])
    gen = [k for k in rank if k in cat] if rank else sorted((k for k in cat if k not in acc and not k.endswith("-access") and cat[k]["jam"] > 0), key=lambda k: -cat[k]["jam"] / cat[k]["cost"])
    todo = acc + [k for k in gen if not acc and y > 1 or cat[k]["cost"] <= cheap]
    out = []
    for k in todo:
        if len(out) < 2 and c.check(out + [{"build": k}])[-1] is None:
            out.append({"build": k})
        elif k.endswith("-access"):
            break
    return out + [{"maintain": maint}]


def run(seed, spill=True, fc=True, maint=2):
    c, rank = City(seed, who="reference"), None
    if fc:
        base = sum(r["delay"] for r in c.forecast({}, 99))
        g = {k: base - sum(r["delay"] for r in c.forecast({0: [{"build": k}]}, 99)) for k in c.menu if not k.endswith("-access")}
        rank = sorted([k for k in g if g[k] > 0], key=lambda k: -g[k] / c.cat[k]["cost"])
    while not c.done:
        c.step(act(c, rank, spill, maint) if c.s["year"] < c.years else [])
    return c


if __name__ == "__main__":
    c = run(int(sys.argv[1]))
    print(c.seed, round(c.score(), 3), [(r["year"], [a["build"] for a in r["actions"] if a.get("ok") and "build" in a]) for r in c.recs if any(a.get("ok") and "build" in a for a in r["actions"])])
