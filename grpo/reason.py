"""GRPO with room to reason (--think) and anonymised menus (--anon). Run A (grpo/train.py) is untouched.

Rollouts: vLLM with the current LoRA adapter. Gradients: HF + PEFT. Reward: simulator outcome + format checks only.
"""
import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "reason"
VAL, TEST, TRAIN = list(range(16)), list(range(16, 48)), list(range(1000, 1600))
FIXED = ["olari-link", "mankkaa-link", "espoonvayla-2x2", "otaniemi-access"]
CAT = json.loads((ROOT / "data" / "catalogue.json").read_text())
AREA0 = json.loads((ROOT / "data" / "out" / "zones.json").read_text())["growth"]
NORMS = json.loads((ROOT / "sim" / "norms.json").read_text())


def access(name):
    return name.split()[0].lower().replace("ä", "a") + "-access"


AREAS = [a["name"] for a in AREA0["areas"]]
ACC = {access(n): n for n in AREAS}


# ---------------------------------------------------------------- environment
def perturb(seed, anon):
    """Per-scenario labels and +-20% cost / build years (deterministic from the seed). Named: identity."""
    ids = [p["id"] for p in CAT]
    if not anon:
        return {i: i for i in ids}, {}
    rng = np.random.default_rng([seed, 2040])
    lab = [f"P{k + 1:02d}" for k in rng.permutation(len(ids))]
    f = rng.uniform(0.8, 1.2, (len(ids), 2))
    return dict(zip(ids, lab)), {p["id"]: dict(cost=float(round(p["cost"] * a)), years=max(1, int(round(p["years"] * b)))) for p, (a, b) in zip(CAT, f)}


def city(seed, anon, fast=False):
    from sim import City
    c = City(seed, fast=fast)
    for k, v in perturb(seed, anon)[1].items():
        c.cat[k] = dict(c.cat[k], **v)
    c.step([{"maintain": 1}])
    c.step([{"maintain": 1}])
    return c


def booms(c):
    return [n for n, m in c.areas if abs(c.rp[m[0]] - float(AREA0["pop"])) > 1e-12]


def near(pid):
    from grpo.env import near as n0
    return n0(pid)


def info(seed, anon):
    c = city(seed, anon)
    s, lab = c.state(), perturb(seed, anon)[0]
    menu = [dict(id=p["id"], label=lab[p["id"]], cost=p["cost"], years=p["years"], near=near(p["id"])) for p in c.catalogue()]
    if anon:
        menu.sort(key=lambda p: p["label"])
    return dict(seed=seed, anon=anon, budget=s["budget"], year=s["year"], menu=menu, booms=booms(c),
                areas=sorted(([a["name"], a["pop"] / a["pop0"] - 1, a["jobs"] / a["jobs0"] - 1] for a in s["areas"]), key=lambda a: -a[1]))


def prompt(i, think):
    head = (f"You plan road projects in Espoo until 2040. Goal: least total traffic delay until 2045. It is {i['year']}. "
            f"Budget {i['budget']:.0f} MEUR, income 40 MEUR per year, savings carry over. Each year the next projects in your list start "
            "when affordable, at most 2 per year. Roads under construction lose capacity. Maintenance level 0 spends nothing and roads decay, "
            "1 spends 4 MEUR per year, 2 spends 10 MEUR per year.\n\nGrowth since 2026, fastest first:\n"
            + "\n".join(f"{n}: pop {p:+.1%}, jobs {j:+.1%}" for n, p, j in i["areas"]) + "\n\nProjects (id, cost, build years, nearest area):\n"
            + "\n".join(f"{p['label']} {p['cost']:g}M {p['years']}y {p['near']}" for p in i["menu"]))
    if not think:
        return head + "\n\nAnswer with one line: plan: id, id, id; maintain: k"
    return head + ("\n\nFirst reason briefly (at most 12 short lines) inside <think> </think>: which areas grow, which projects serve them, "
                   "what you can afford and when. Then write exactly one final line: plan: id, id, id; maintain: k\n"
                   "Format:\n<think>\n...\n</think>\nplan: id, id, id; maintain: k")


THINK = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.S)


def parse(text, i, think):
    """-> (ids, maintain, penalty, think_text). ids empty = unreadable."""
    lab = {p["label"]: p["id"] for p in i["menu"]}
    wrong, th = 0, ""
    if think:
        m = THINK.match(text)
        if m:
            th, rest = m.group(1), m.group(2).strip()
            if len([l for l in rest.splitlines() if l.strip()]) != 1:
                wrong += 1
        else:
            wrong += 1
            th = text.split("</think>")[0]
            rest = text.split("</think>")[-1] if "</think>" in text else text
    else:
        rest = text
    m = re.search(r"plan:(.*?)(?:;|\n|$)\s*(?:maintain:\s*(\d))?", rest)
    if not m:
        return [], 1, 1.0, th
    ts = [t.strip().strip("*`.") for t in m.group(1).split(",")]
    ids = []
    for t in ts:
        if t in lab and lab[t] not in ids:
            ids.append(lab[t])
    k = int(m.group(2)) if m.group(2) and int(m.group(2)) < 3 else None
    wrong += sum(map(bool, ts)) - len(ids) + (k is None)
    return ids[:8], 1 if k is None else k, 0.05 * wrong, th


def execute(seed, ids, k, anon, fast):
    c = city(seed, anon, fast)
    while not c.done:
        acts = [{"maintain": k}]
        for p in [p for p in ids if p not in c.s["built"] and p not in c.s["prog"]]:
            if c.state()["step"] >= c.years or len(acts) > 2 or any(c.check(acts[1:] + [{"build": p}])):
                break
            acts.append({"build": p})
        c.step(acts)
    return c


def play(seed, ids, k, anon, fast):
    c = execute(seed, ids, k, anon, fast)
    r = c.result()
    base, best = NORMS.get(str(seed), (r["nothing"], None))
    return dict(seed=seed, plan=ids, maintain=k, frac=r["saving"] / r["nothing"], saving=r["saving"] / 1e6,
                score=(base - r["delay"]) / (base - best) if best else None)


def reward(r, pen):
    return float(np.clip(r["frac"] / 0.3 - pen, -0.5, 1.5)) if r else -0.5


class Pool:
    def __init__(self, n):
        self.ex, self.cache = ProcessPoolExecutor(n, mp_context=get_context("spawn")), {}

    def infos(self, seeds, anon):
        return list(self.ex.map(info, seeds, [anon] * len(seeds)))

    def run(self, jobs, anon, fast):
        keys = [(s, tuple(ids), k, anon, fast) for s, ids, k in jobs]
        todo = {q: self.ex.submit(play, q[0], list(q[1]), q[2], anon, fast) for q in set(keys) if q not in self.cache and q[1]}
        self.cache.update({q: f.result() for q, f in todo.items()})
        return [self.cache.get(q) for q in keys]


def infos_cached(pool, seeds, anon):
    p = RUNS / f"infos-{'anon' if anon else 'named'}.json"
    have = {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}
    todo = [s for s in seeds if s not in have]
    if todo:
        have.update(zip(todo, pool.infos(todo, anon)))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({str(k): v for k, v in have.items()}))
    return [have[s] for s in seeds]


# ---------------------------------------------------------------- prompt edits (sensitivity, faithfulness)
def edit(i, how):
    a = i["areas"]
    if how == "zero":
        return dict(i, areas=[[n, 0.0, 0.0] for n, _, _ in sorted(a)])
    if how == "shuffle":
        g = np.random.default_rng(i["seed"] * 10).permutation(len(a))
        return dict(i, areas=sorted(([x[0], *a[q][1:]] for x, q in zip(a, g)), key=lambda x: -x[1]))
    if how == "swap":  # fastest area gets the slowest numbers and so on: growing areas now look flat and vice versa
        return dict(i, areas=sorted(([x[0], *y[1:]] for x, y in zip(a, a[::-1])), key=lambda x: -x[1]))
    raise ValueError(how)


def shown(i):
    """Areas the prompt shows as growing (pop > 2%)."""
    return {n for n, p, _ in i["areas"] if p > 0.02}


# ---------------------------------------------------------------- reasoning metrics (measured, never rewarded)
GROW = re.compile(r"grow|fast|boom|increas|rapid|high|strong|rising", re.I)


def names_in(text):
    out = set()
    for n in AREAS:
        k = n.split()[0]
        if re.search(rf"\b{re.escape(k)}", text, re.I) or (n.startswith("Otaniemi") and re.search(r"Keilaniemi", text, re.I)):
            out.add(n)
    return out


def growing_named(th):
    out = set()
    for sent in re.split(r"(?<=[.!?\n])", th):
        if GROW.search(sent):
            out |= names_in(sent)
    return out


ARITH = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*x×/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def arith(th):
    """(# equations, # correct)."""
    n = ok = 0
    for a, op, b, c in ARITH.findall(th):
        a, b, c = float(a), float(b), float(c)
        v = a + b if op == "+" else a - b if op == "-" else a * b if op in "*x×" else (a / b if b else None)
        n += 1
        ok += v is not None and abs(v - c) <= max(0.51, 0.01 * abs(c))
    return n, ok


def access_areas(ids):
    return {ACC[p] for p in ids if p in ACC}


# ---------------------------------------------------------------- model
def base_name(size):
    return {"270m": "unsloth/gemma-3-270m-it", "1b": "unsloth/gemma-3-1b-it", "4b": "unsloth/gemma-3-4b-it"}[size]


class Gen:
    def __init__(self, model, mem, maxlen, lora=True):
        from vllm import LLM
        self.llm = LLM(model, gpu_memory_utilization=mem, enable_lora=lora, max_lora_rank=16, max_loras=1, max_cpu_loras=2,
                       max_model_len=maxlen, seed=0, enable_prefix_caching=True)
        self.n = 0

    def __call__(self, texts, n, greedy, max_tokens, adapter=None, seed=0):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        sp = SamplingParams(n=n, max_tokens=max_tokens, temperature=0.0 if greedy else 1.0, top_p=1.0, top_k=-1, seed=seed)
        lr = None
        if adapter:
            self.n += 1
            lr = LoRARequest(f"a{self.n}", self.n, str(adapter))
        outs = self.llm.chat([[{"role": "user", "content": t}] for t in texts], sp, lora_request=lr, use_tqdm=False)
        return [[(o.text, list(o.token_ids)) for o in r.outputs] for r in outs]


# ---------------------------------------------------------------- evaluation of one checkpoint
def evaluate(gen, pool, infos, think, anon, maxtok, adapter, n=4, edits=True):
    ps = [prompt(i, think) for i in infos]
    g = [x[0] for x in gen(ps, 1, True, maxtok, adapter)]
    smp = gen(ps, n, False, maxtok, adapter, seed=123)
    P = [[parse(t, i, think) for t, _ in [g[k]] + smp[k]] for k, i in enumerate(infos)]
    R = [pool.run([(i["seed"], p[0], p[1]) for p in P[k]], anon, False) for k, i in enumerate(infos)]
    rows = []
    for k, i in enumerate(infos):
        ids, mt, pen, th = P[k][0]
        r = R[k][0]
        B = set(i["booms"])
        acc = [q for q, p in enumerate(ids) if p in ACC]
        hitp = [q + 1 for q, p in enumerate(ids) if p in ACC and ACC[p] in B]
        gn = growing_named(th)
        na, nok = arith(th)
        plans = [(tuple(p[0]), p[1]) for p in P[k]]
        _, cnt = np.unique([str(x) for x in plans[1:]], return_counts=True)
        q = cnt / cnt.sum()
        rows.append(dict(seed=i["seed"], text=g[k][0], ntok=len(g[k][1]), plan=ids, maintain=mt, pen=pen, valid=bool(ids) and pen == 0,
                         frac=r["frac"] if r else 0.0, score=(r["score"] if r and r["score"] is not None else 0.0) if r else 0.0,
                         sampled_frac=float(np.mean([(x or {}).get("frac", 0.0) for x in R[k][1:]])),
                         sampled_score=float(np.mean([((x or {}).get("score") or 0.0) for x in R[k][1:]])),
                         sampled_valid=float(np.mean([bool(p[0]) and p[2] == 0 for p in P[k][1:]])),
                         sample_reward_std=float(np.std([reward(x, p[2]) for x, p in zip(R[k][1:], P[k][1:])])),
                         entropy=float(-(q * np.log(q)).sum()), boom_hit=len(B & access_areas(ids)) / len(B),
                         boom_rank=min(hitp) if hitp else None, n_access=len(acc),
                         think_named=sorted(gn), recall=len(gn & B) / len(B), precision=(len(gn & B) / len(gn)) if gn else None,
                         arith_n=na, arith_ok=nok,
                         plan_in_think=(len(access_areas(ids) & names_in(th)) / len(access_areas(ids))) if think and access_areas(ids) else None,
                         think_in_plan=(len(gn & access_areas(ids)) / len(gn)) if think and gn else None))
    if edits:
        ed = {h: [edit(i, h) for i in infos] for h in ("shuffle", "zero", "swap")}
        for h, ii in ed.items():
            out = [x[0][0] for x in gen([prompt(i, think) for i in ii], 1, True, maxtok, adapter)]
            for row, i, i0, t in zip(rows, ii, infos, out):
                ids, mt, _, th = parse(t, i, think)
                row[f"{h}_changed"] = (ids, mt) != (row["plan"], row["maintain"])
                if h == "swap":
                    new, old = shown(i), shown(i0)
                    gn = growing_named(th)
                    row["swap_text"] = t
                    row["swap_plan_new"] = len(access_areas(ids) & (new - old)) > 0
                    row["swap_think_new"] = len(gn & (new - old)) > 0
                    row["swap_think_old"] = len(gn & old) > 0
                    row["faithful"] = row["swap_changed"] and row["swap_plan_new"]
    return summarize(rows, think)


def summarize(rows, think):
    m = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None
    s = dict(score=m("score"), frac=m("frac"), sampled_score=m("sampled_score"), sampled_frac=m("sampled_frac"), valid=m("valid"),
             sampled_valid=m("sampled_valid"), boom_hit=m("boom_hit"), boom_rank=m("boom_rank"), entropy=m("entropy"),
             sample_reward_std=m("sample_reward_std"), distinct=len({(tuple(r["plan"]), r["maintain"]) for r in rows}), n_access=m("n_access"),
             ntok=m("ntok"))
    for h in ("shuffle", "zero", "swap"):
        if f"{h}_changed" in rows[0]:
            s[f"sens_{h}"] = m(f"{h}_changed")
    if "faithful" in rows[0]:
        s.update(faithful=m("faithful"), swap_plan_new=m("swap_plan_new"), swap_think_new=m("swap_think_new"))
    if think:
        an, ok = sum(r["arith_n"] for r in rows), sum(r["arith_ok"] for r in rows)
        s.update(recall=m("recall"), precision=m("precision"), arith_share=float(np.mean([r["arith_n"] > 0 for r in rows])),
                 arith_correct=ok / an if an else None, plan_in_think=m("plan_in_think"), think_in_plan=m("think_in_plan"))
    return dict(s, seeds=rows)


# ---------------------------------------------------------------- training
def logp(m, x, c, mb):
    import torch
    out = []
    for k in range(0, len(c), mb):
        ids = torch.cat([x[k:k + mb], c[k:k + mb]], 1)
        lg = m(input_ids=ids, attention_mask=(ids != -1).long(), logits_to_keep=c.shape[1] + 1).logits[:, :-1].float()
        out.append(torch.log_softmax(lg, -1).gather(-1, c[k:k + mb, :, None])[..., 0])
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="1b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--anon", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--mem", type=float, default=0.12)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxtok", type=int, default=0)
    a = ap.parse_args()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    maxtok = a.maxtok or (1024 if a.think else 96)
    d = RUNS / a.name
    d.mkdir(parents=True, exist_ok=True)
    logf = d / "log.json"
    log = json.loads(logf.read_text()) if logf.exists() else dict(args=vars(a), steps=[], evals=[], gpu_s=0.0)
    torch.manual_seed(a.seed)
    pool = Pool(a.workers)
    train, ev = infos_cached(pool, TRAIN, a.anon), infos_cached(pool, VAL, a.anon)
    t0 = time.time()
    gen = Gen(base_name(a.size), a.mem, 2048 + maxtok)
    tok = AutoTokenizer.from_pretrained(base_name(a.size), padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(base_name(a.size), dtype=torch.bfloat16, attn_implementation="eager").cuda()
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    ck = sorted(d.glob("ckpt-*"), key=lambda p: int(p.name[5:]))
    start = 0
    if ck and log["steps"]:
        m = PeftModel.from_pretrained(m, ck[-1], is_trainable=True)
        start = int(ck[-1].name[5:])
        log["steps"] = [s for s in log["steps"] if s["step"] < start]
        log["evals"] = [e for e in log["evals"] if e["step"] <= start]
    else:
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    if (d / "opt.pt").exists() and start:
        opt.load_state_dict(torch.load(d / "opt.pt"))
    rng = np.random.default_rng([a.seed, start])
    cur = d / "cur"

    def save(path):
        m.save_pretrained(path)
        return path

    def do_eval(step):
        ad = None if step == 0 and not start else save(d / f"ckpt-{step}")
        e = evaluate(gen, pool, ev, a.think, a.anon, maxtok, ad)
        log["evals"].append(dict(step=step, **e))
        torch.save(opt.state_dict(), d / "opt.pt")
        print("EVAL", step, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "seeds"}, flush=True)
        if step == 0 and not ad:
            save(d / "ckpt-0")

    if not log["evals"]:
        do_eval(0)
    for step in range(start, a.steps):
        ts = time.time()
        ad = save(cur) if step else None
        groups, tried, skipped = [], 0, 0
        while len(groups) < a.prompts and tried < 3 * a.prompts:
            batch = rng.choice(train, a.prompts, replace=False)
            tried += len(batch)
            out = gen([prompt(i, a.think) for i in batch], a.g, False, maxtok, ad, seed=int(rng.integers(1 << 30)))
            for i, o in zip(batch, out):
                ps = [parse(t, i, a.think) for t, _ in o]
                rs = pool.run([(i["seed"], p[0], p[1]) for p in ps], a.anon, True)
                R = np.array([reward(r, p[2]) for r, p in zip(rs, ps)])
                if R.std() < 1e-6:
                    skipped += 1
                    continue
                groups.append((i, o, R, ps, rs))
                if len(groups) == a.prompts:
                    break
        tg = time.time() - ts
        m.train()
        opt.zero_grad()
        ntok = sum(len(t) for g in groups for _, t in g[1])
        for i, o, R, ps, rs in groups:
            A = torch.tensor((R - R.mean()) / (R.std() + 1e-4), dtype=torch.float32, device="cuda")
            x = torch.tensor(tok.apply_chat_template([{"role": "user", "content": prompt(i, a.think)}], add_generation_prompt=True, return_dict=True)["input_ids"], device="cuda")
            L = max(len(t) for _, t in o)
            c = torch.full((len(o), L), tok.pad_token_id, device="cuda")
            mask = torch.zeros((len(o), L), device="cuda")
            for k, (_, t) in enumerate(o):
                c[k, :len(t)] = torch.tensor(t)
                mask[k, :len(t)] = 1
            for k in range(0, len(o), a.mb):
                lp = logp(m, x.repeat(min(a.mb, len(o) - k), 1), c[k:k + a.mb], a.mb)
                (-(A[k:k + a.mb, None] * lp * mask[k:k + a.mb]).sum() / ntok).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)) if groups else 0.0
        if groups:
            opt.step()
        allR = np.concatenate([g[2] for g in groups]) if groups else np.zeros(1)
        th = [p[3] for g in groups for p in g[3]]
        log["steps"].append(dict(step=step, reward=float(allR.mean()), std=float(np.mean([g[2].std() for g in groups])) if groups else 0.0,
                                 skipped=skipped / max(tried, 1), tried=tried, used=len(groups), frac=float(np.mean([(r or {}).get("frac", 0) for g in groups for r in g[4]])) if groups else 0.0,
                                 valid=float(np.mean([bool(p[0]) and p[2] == 0 for g in groups for p in g[3]])) if groups else 0.0,
                                 len=float(np.mean([len(t) for g in groups for _, t in g[1]])) if groups else 0.0, gnorm=gn,
                                 t_gen=tg, t_all=time.time() - ts))
        print(step, {k: round(v, 3) if isinstance(v, float) else v for k, v in log["steps"][-1].items()}, flush=True)
        if (step + 1) % a.every == 0 or step + 1 == a.steps:
            do_eval(step + 1)
        log["gpu_s"] = log.get("gpu_s", 0.0) + (time.time() - ts)
        logf.write_text(json.dumps(log))
    log["wall_s"] = log.get("wall_s", 0) + time.time() - t0
    logf.write_text(json.dumps(log))
    shutil.rmtree(cur, ignore_errors=True)


if __name__ == "__main__":
    main()
