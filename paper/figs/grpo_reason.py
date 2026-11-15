"""GRPO reasoning experiments (grpo/reason.py): summary data, figures, tables, trace excerpts and macros.

  python paper/figs/grpo_reason.py          # collect runs/reason -> data/grpo_reason.json (+ evidence traces), then build
make.py calls build(M) so the macros land in numbers.tex; build only needs the committed data file.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "reason"
DATA = HERE / "data" / "grpo_reason.json"
TRACE_SEEDS = [0, 1]  # fixed before looking at any output
RATE_EUR_H = 1.70  # assumed Verda on-demand price for 1x RTX PRO 6000 (EUR per GPU hour); replace with the invoice rate
RUNS_ORDER = [("1b-think-anon", "ThinkAnon", "1B think + anon"), ("1b-think-named", "ThinkNamed", "1B think + named"),
              ("1b-nothink-anon", "NothinkAnon", "1B no-think + anon"), ("1b-nothink-named", "NothinkNamed", "1B no-think + named"),
              ("4b-think-anon", "FourThinkAnon", "4B think + anon")]
KEEP = ("score", "frac", "sampled_score", "sampled_frac", "valid", "sampled_valid", "boom_hit", "boom_rank", "entropy", "sample_reward_std", "distinct",
        "n_access", "ntok", "sens_shuffle", "sens_zero", "sens_swap", "faithful", "swap_plan_new", "swap_think_new", "recall", "precision",
        "arith_share", "arith_correct", "plan_in_think", "think_in_plan")
SEEDKEEP = ("seed", "score", "frac", "boom_hit", "plan", "maintain", "valid", "ntok", "recall")


def boot(x, n=10000):
    x = np.asarray(x, float)
    m = np.random.default_rng(0).choice(x, (n, x.size)).mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(x.size))


def slim(e):
    return dict({k: e.get(k) for k in KEEP}, step=e.get("step"), seeds=[{k: s.get(k) for k in SEEDKEEP} for s in e["seeds"]])


def collect():
    out = dict(runs={}, refs={}, zero={}, test={})
    for name, _, _ in RUNS_ORDER:
        f = RUNS / name / "log.json"
        if not f.exists():
            continue
        log = json.loads(f.read_text())
        out["runs"][name] = dict(args=log["args"], steps=log["steps"], evals=[slim(e) for e in log["evals"]], gpu_s=log.get("gpu_s"), wall_s=log.get("wall_s"))
        ev = log["evals"]
        picks = sorted({ev[0]["step"], ev[len(ev) // 2]["step"], ev[-1]["step"]})
        d = ROOT / "evidence" / "grpo-reasoning" / name
        d.mkdir(parents=True, exist_ok=True)
        for e in ev:
            if e["step"] in picks:
                for s in e["seeds"]:
                    if s["seed"] in TRACE_SEEDS:
                        (d / f"{s['seed']}_{e['step']}.txt").write_text(
                            f"run {name}, step {e['step']}, validation seed {s['seed']} (greedy)\nplan executed: {s['plan']} maintain {s['maintain']}\n"
                            f"score {s['score']:.3f}, saved share {s['frac']:.4f}, boom-hit {s['boom_hit']:.2f}\n\n{s['text']}\n")
    ev = RUNS / "eval"
    for k in ("refs", "zero", "test"):
        if (ev / f"{k}.json").exists():
            d = json.loads((ev / f"{k}.json").read_text())
            out[k] = d if k == "refs" else {kk: slim(v) for kk, v in d.items()}
    DATA.write_text(json.dumps(out))


def fmt(x, n=2):
    return "--" if x is None else f"{x:.{n}f}"


def build(M):
    if not DATA.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = json.loads(DATA.read_text())
    runs = {n: D["runs"][n] for n, _, _ in RUNS_ORDER if n in D["runs"]}
    tag = {n: t for n, t, _ in RUNS_ORDER}
    lab = {n: l for n, _, l in RUNS_ORDER}
    col = {"1b-think-anon": "#c2185b", "1b-think-named": "#f48fb1", "1b-nothink-anon": "#1565c0", "1b-nothink-named": "#90caf9", "4b-think-anon": "#6a1b9a"}
    grd = json.loads((HERE / "data" / "grpo.json").read_text())
    norms = json.loads((ROOT / "sim" / "norms.json").read_text())

    def best(r):
        return max(r["evals"], key=lambda e: (e["score"], e["step"]))

    # --- macros per run
    hours_total = 0.0
    for n, r in runs.items():
        t, z, e0, b = tag[n], r["evals"][-1], r["evals"][0], best(r)
        sc = boot([s["score"] for s in z["seeds"]])
        M.update({f"grpo{t}Score": fmt(sc["mean"]), f"grpo{t}ScoreLo": fmt(sc["lo"]), f"grpo{t}ScoreHi": fmt(sc["hi"]),
                  f"grpo{t}ScoreZero": fmt(e0["score"]), f"grpo{t}ScoreBest": fmt(b["score"]), f"grpo{t}BestStep": str(b["step"]),
                  f"grpo{t}Steps": str(z["step"]), f"grpo{t}Frac": fmt(100 * z["frac"], 1),
                  f"grpo{t}BoomHit": fmt(z["boom_hit"]), f"grpo{t}BoomHitZero": fmt(e0["boom_hit"]), f"grpo{t}Sens": fmt(z.get("sens_shuffle")),
                  f"grpo{t}SensZero": fmt(z.get("sens_zero")), f"grpo{t}Distinct": str(z["distinct"]), f"grpo{t}DistinctZero": str(e0["distinct"]),
                  f"grpo{t}Valid": fmt(z["valid"]), f"grpo{t}ValidZero": fmt(e0["valid"]), f"grpo{t}Faithful": fmt(z.get("faithful")),
                  f"grpo{t}FaithfulZero": fmt(e0.get("faithful")), f"grpo{t}Skipped": fmt(float(np.mean([s["skipped"] for s in r["steps"]]))),
                  f"grpo{t}GpuHours": fmt(r["gpu_s"] / 3600 if r.get("gpu_s") else None, 2)})
        if r["args"]["think"]:
            M.update({f"grpo{t}ThinkRecall": fmt(z.get("recall")), f"grpo{t}ThinkRecallZero": fmt(e0.get("recall")),
                      f"grpo{t}ThinkPrecision": fmt(z.get("precision")), f"grpo{t}ThinkLen": f"{z['ntok']:.0f}", f"grpo{t}ThinkLenZero": f"{e0['ntok']:.0f}",
                      f"grpo{t}Arith": fmt(z.get("arith_share")), f"grpo{t}Consist": fmt(z.get("think_in_plan"))})
        hours_total += (r.get("gpu_s") or 0) / 3600
        for k, e in (("Final", z), ("Best", b)):
            te = D["test"].get(f"{n}|{k.lower()}|{e['step']}")
            if te:
                tb = boot([s["frac"] for s in te["seeds"]])
                ts = boot([s["score"] for s in te["seeds"]])
                M.update({f"grpo{t}Test" + ("" if k == "Final" else "Best"): fmt(100 * tb["mean"], 1),
                          f"grpo{t}TestScore" + ("" if k == "Final" else "Best"): fmt(ts["mean"]),
                          f"grpo{t}TestBoomHit" + ("" if k == "Final" else "Best"): fmt(te["boom_hit"])})
    # --- reference points
    for k, v in D["refs"].items():
        nm = "".join(w.capitalize() for w in k.split("-"))
        for sp, ss in (("Val", range(16)), ("Test", range(16, 48))):
            f = boot([v[str(s)]["frac"] for s in ss])
            M[f"grpoRef{nm}{sp}"] = fmt(100 * f["mean"], 1)
            if sp == "Val":
                M[f"grpoRef{nm}Score"] = fmt(float(np.mean([v[str(s)]["score"] for s in ss])))
                M[f"grpoRef{nm}BoomHit"] = fmt(float(np.mean([v[str(s)]["boom_hit"] for s in ss])))
    for k, v in D["zero"].items():
        nm = "Zero" + "".join(w.capitalize() for w in k.split("-")).replace("1b", "One").replace("4b", "Four").replace("1B", "One").replace("4B", "Four")
        M[nm + "Score"] = fmt(v["score"])
        M[nm + "Frac"] = fmt(100 * v["frac"], 1)
        M[nm + "Valid"] = fmt(v["valid"])
        M[nm + "BoomHit"] = fmt(v["boom_hit"])
        if v.get("recall") is not None:
            M[nm + "Recall"] = fmt(v["recall"])
    M["grpoReasonGpuHours"] = fmt(hours_total, 1)
    M["grpoGpuRate"] = f"{RATE_EUR_H:.2f}"
    # run A aliases asked for by the experiment plan
    for a, b in (("grpoConst", "grpoConstVal"), ("grpoSensWarm", "grpoSensWarmShuffle"), ("grpoSensFinal", "grpoSensFinalShuffle")):
        if b in M:
            M[a] = M[b]

    # --- figure: curves
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9))
    A = grd["strategy"]
    ax[0].plot([r["step"] for r in A], [r["score"] for r in A], color="#9e9e9e", lw=1, ls="--", label="run A (270M)")
    ax[1].plot([r["step"] for r in A], [r["boom_hit"] for r in A], color="#9e9e9e", lw=1, ls="--")
    ax[2].plot([r["step"] for r in A], [r["sample_distinct"] for r in A], color="#9e9e9e", lw=1, ls="--")
    ax[3].plot([r["step"] for r in A if r["train_std"] is not None], [r["train_std"] for r in A if r["train_std"] is not None], color="#9e9e9e", lw=1, ls="--")
    for n, r in runs.items():
        x = [e["step"] for e in r["evals"]]
        ax[0].plot(x, [e["score"] for e in r["evals"]], color=col[n], lw=1.2, marker="o", ms=2, label=lab[n])
        ax[1].plot(x, [e["boom_hit"] for e in r["evals"]], color=col[n], lw=1.2, marker="o", ms=2)
        ax[2].plot(x, [e["distinct"] for e in r["evals"]], color=col[n], lw=1.2, marker="o", ms=2)
        st = r["steps"]
        w = 5
        ax[3].plot([s["step"] for s in st][w - 1:], np.convolve([s["std"] for s in st], np.ones(w) / w, "valid"), color=col[n], lw=1.2)
    for a_, t in zip(ax, ("val score (greedy)", "boom-hit", "distinct plans (16 seeds)\nrun A: per-seed samples", "within-group reward std\n(used groups, 5-step mean)")):
        a_.set_title(t)
        a_.set_xlabel("GRPO step")
    ax[0].legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "grpo_curves.pdf")
    plt.close(fig)

    # --- figure: reasoning
    th = {n: r for n, r in runs.items() if r["args"]["think"]}
    if th:
        keys = (("ntok", "think length (tokens)"), ("recall", "growth recall in <think>"), ("arith_share", "traces with arithmetic"),
                ("think_in_plan", "named growing area\nupgraded in plan"), ("faithful", "faithfulness (swap test)"))
        fig, ax = plt.subplots(1, len(keys), figsize=(7.0, 1.8))
        for a_, (k, t) in zip(ax, keys):
            for n, r in th.items():
                a_.plot([e["step"] for e in r["evals"]], [e.get(k) if e.get(k) is not None else np.nan for e in r["evals"]], color=col[n], lw=1.2, marker="o", ms=2, label=lab[n])
            a_.set_title(t)
            a_.set_xlabel("GRPO step")
        ax[0].legend(fontsize=5.5)
        fig.tight_layout()
        fig.savefig(HERE / "grpo_reasoning.pdf")
        plt.close(fig)

    # --- table: all runs and baselines (final and best-validation)
    rows = []
    for n, r in runs.items():
        for k, e in (("start", r["evals"][0]), ("final", r["evals"][-1]), ("best val", best(r))):
            sc = boot([s["score"] for s in e["seeds"]])
            te = D["test"].get(f"{n}|{'best' if k == 'best val' else 'final'}|{e['step']}") if k != "start" else D["zero"].get(
                "1b-" + n.split("-", 1)[1] + "-test")
            tf = boot([s["frac"] for s in te["seeds"]]) if te else None
            rows.append(f"{lab[n]}, {k} (step {e['step']}) & {sc['mean']:.2f} [{sc['lo']:.2f}, {sc['hi']:.2f}] & {100 * e['frac']:.1f} & "
                        + (f"{100 * tf['mean']:.1f} [{100 * tf['lo']:.1f}, {100 * tf['hi']:.1f}]" if tf else "--")
                        + f" & {e['valid']:.2f} & {e['boom_hit']:.2f} \\\\")
    for k, v in sorted(D["refs"].items()):
        sv, st = boot([v[str(s)]["score"] for s in range(16)]), boot([v[str(s)]["frac"] for s in range(16, 48)])
        nm = {"oracle": "Growth-aware oracle list (hidden info)", "random": "Random valid lists (4 per seed)", "const": "Run A constant plan"}[k.split("-")[0]]
        rows.append(f"{nm}, {k.split('-')[1]} & {sv['mean']:.2f} [{sv['lo']:.2f}, {sv['hi']:.2f}] & {100 * np.mean([v[str(s)]['frac'] for s in range(16)]):.1f} & "
                    f"{100 * st['mean']:.1f} [{100 * st['lo']:.1f}, {100 * st['hi']:.1f}] & 1.00 & {np.mean([v[str(s)]['boom_hit'] for s in range(16)]):.2f} \\\\")
    for k, v in sorted(D["zero"].items()):
        if k.endswith("-val") and k.startswith("4b"):
            rows.append(f"Gemma 3 4B zero-shot, {k[3:-4].replace('-', ' + ')} & {v['score']:.2f} & {100 * v['frac']:.1f} & -- & {v['valid']:.2f} & {v['boom_hit']:.2f} \\\\")
    (HERE / "grpo_runs.tex").write_text("\n".join(rows) + "\n")

    # --- table: factorial
    fr = []
    for n in ("1b-think-anon", "1b-think-named", "1b-nothink-anon", "1b-nothink-named"):
        if n in runs:
            e0, z = runs[n]["evals"][0], runs[n]["evals"][-1]
            fr.append(f"{lab[n][3:]} & {e0['score']:.2f} $\\to$ {z['score']:.2f} & {e0['boom_hit']:.2f} $\\to$ {z['boom_hit']:.2f} & "
                      f"{fmt(e0.get('sens_shuffle'))} $\\to$ {fmt(z.get('sens_shuffle'))} & {e0['distinct']} $\\to$ {z['distinct']} & "
                      f"{fmt(e0.get('faithful'))} $\\to$ {fmt(z.get('faithful'))} \\\\")
    (HERE / "grpo_factorial.tex").write_text("\n".join(fr) + "\n")

    # --- table: trace excerpts (seeds fixed in advance; first 160 chars of the reasoning + the plan line)
    tr = []
    ex = ROOT / "evidence" / "grpo-reasoning" / "1b-think-anon"
    if ex.exists():
        for f in sorted(ex.glob("*.txt"), key=lambda p: tuple(map(int, p.stem.split("_")))):
            txt = f.read_text().split("\n\n", 1)[1]
            body = re.sub(r"\s+", " ", txt.split("</think>")[0].replace("<think>", "")).strip()
            plan = re.findall(r"plan:[^\n]*", txt)
            esc = lambda s: s.replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$").replace("*", "")
            sd, stp = f.stem.split("_")
            tr.append(f"{sd} & {stp} & \\emph{{{esc(body[:170])}\\dots}} & \\texttt{{{esc(plan[-1][:60]) if plan else '(none)'}}} \\\\")
    (HERE / "grpo_trace.tex").write_text("\n".join(tr) + "\n")


if __name__ == "__main__":
    if RUNS.exists() and "--no-collect" not in sys.argv:
        collect()
    M = {}
    build(M)
    for k, v in sorted(M.items()):
        print(f"\\{k} = {v}")
