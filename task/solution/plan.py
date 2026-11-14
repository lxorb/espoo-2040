RANK = ["espoonvayla-2x2", "kauklahdenvayla-2x2", "olari-link", "keha2-turunvayla-junction", "mankkaa-link", "keilaniemi-junction",
        "kivenlahti-bridge", "espoonvayla-mid", "turunvayla-east-lanes"]


def decide(st, rows, late=False, cheap=15, maint=2):
    y, ok = st["step"], set(st["startable"])
    cat = {r[0]: dict(name=r[1], cost=r[3], jam=r[7]) for r in rows if r[6] == "open"}
    grow = {a["name"].split(" and ")[0]: (a["pop"] + a["jobs"]) / (a["pop0"] + a["jobs0"]) - 1 for a in st["areas"]}
    acc = [k for k in cat if k.endswith("-access")]
    boom = sorted({k for n in grow if y > 1 and grow[n] > 0.015 * (y - 1) for k in acc if cat[k]["name"].startswith(n)}, key=lambda k: -cat[k]["jam"])
    top = [k for k in sorted(acc, key=lambda k: -cat[k]["jam"])[:1] if boom and k not in boom]
    todo = (boom + top if late else top + boom)
    todo += [k for k in RANK if k in cat and (not todo and y > 1 or cat[k]["cost"] <= cheap)]
    out = []
    for k in todo:
        if len(out) < 2 and k in ok:
            out.append(k)
        elif k.endswith("-access"):
            break
    return [{"build": k} for k in out] + [{"maintain": maint}]
