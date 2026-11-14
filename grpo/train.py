import argparse
import json
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import OUT, Pool, parse, prompt, reward

BASE, MAXN, EVAL = "unsloth/gemma-3-270m-it", 64, list(range(16))


def load(ckpt=None, r=16):
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    m.generation_config.max_length = None
    if ckpt:
        return tok, PeftModel.from_pretrained(m, ckpt, is_trainable=True)
    return tok, get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))


def enc(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]


def sample(m, tok, texts, n, greedy=False, temp=1.0):
    m.eval()
    x = tok.apply_chat_template([[{"role": "user", "content": t}] for t in texts for _ in range(n)], add_generation_prompt=True, padding=True,
                                return_tensors="pt", return_dict=True)
    with torch.no_grad():
        o = m.generate(**x, max_new_tokens=MAXN, do_sample=not greedy, temperature=temp, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
    c = o[:, x["input_ids"].shape[1]:]
    eos = m.generation_config.eos_token_id
    end = torch.isin(c, torch.tensor(eos if isinstance(eos, list) else [eos]))
    return c, (end.cumsum(1) - end.int()) == 0, [tok.decode(y, skip_special_tokens=True) for y in c]


def logp(m, ids, c, temp=1.0):
    m.train()
    x = torch.cat([ids.repeat(len(c), 1), c], 1)
    lg = m(input_ids=x, logits_to_keep=c.shape[1] + 1).logits[:, :-1].float() / temp
    return torch.log_softmax(lg, -1).gather(-1, c[..., None])[..., 0]


def evaluate(m, tok, pool, infos, n=4, off=False):
    with m.disable_adapter() if off else torch.no_grad():
        ps = [prompt(i) for i in infos]
        g, r = sample(m, tok, ps, 1, True)[2], sample(m, tok, ps, n)[2]
    texts = [[g[k]] + r[k * n:(k + 1) * n] for k in range(len(infos))]
    jobs = [(i["seed"], *parse(t, [p["id"] for p in i["menu"]])) for i, ts in zip(infos, texts) for t in ts]
    rs = pool.run([j[:3] for j in jobs])
    out = []
    for k, i in enumerate(infos):
        g = list(zip(jobs[k * (n + 1):(k + 1) * (n + 1)], rs[k * (n + 1):(k + 1) * (n + 1)]))
        sc = [r["score"] if r else None for _, r in g]
        out.append(dict(seed=i["seed"], text=texts[k][0], greedy=g[0][1], score=sc[0] or 0.0, sampled=float(np.mean([s or 0.0 for s in sc[1:]])),
                        valid=float(np.mean([bool(j[1]) and j[3] == 0 for j, _ in g]))))
    f = lambda k: float(np.mean([o[k] for o in out]))
    return dict(score=f("score"), sampled=f("sampled"), valid=f("valid"),
                saving=float(np.mean([o["greedy"]["saving"] if o["greedy"] else 0 for o in out])), seeds=out)


def warm(m, tok, infos, steps, lr, rng):
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    for s in range(steps):
        opt.zero_grad()
        for i in rng.choice(infos, 4):
            ids = [p["id"] for p in i["menu"]]
            t = f"plan: {', '.join(rng.permutation(ids)[:rng.integers(2, 7)])}; maintain: {rng.integers(3)}"
            c = tok(t + "<end_of_turn>", add_special_tokens=False, return_tensors="pt")["input_ids"]
            (-logp(m, enc(tok, prompt(i)), c).mean() / 4).backward()
        opt.step()
        if s % 20 == 0:
            print("warm", s, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=120)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--ckpt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    rng, pool = np.random.default_rng(0), Pool(a.workers)
    tok, m = load(a.ckpt)
    train, ev = pool.infos(range(1000, 1000 + a.pool)), pool.infos(EVAL)
    log = json.loads((OUT / "curve.json").read_text()) if (OUT / "curve.json").exists() else dict(args=vars(a), steps=[], evals=[])
    if not a.ckpt:
        log["evals"].append(dict(step=-1, name="zero-shot", **evaluate(m, tok, pool, ev, off=True)))
        warm(m, tok, train, a.warm, 2e-4, rng)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    s0 = log["steps"][-1]["step"] + 1 if log["steps"] else 0
    for step in range(s0, s0 + a.steps):
        if step % a.every == 0:
            e = evaluate(m, tok, pool, ev)
            log["evals"].append(dict(step=step, name="grpo" if step else "warm", **e))
            m.save_pretrained(OUT / f"ckpt-{step}")
            print("eval", step, round(e["score"], 3), round(e["sampled"], 3), round(e["valid"], 3), flush=True)
        t0 = time.time()
        batch = rng.choice(train, a.prompts, replace=False)
        c, mask, texts = sample(m, tok, [prompt(i) for i in batch], a.g, temp=a.temp)
        groups = [(i, enc(tok, prompt(i)), c[k * a.g:(k + 1) * a.g], mask[k * a.g:(k + 1) * a.g],
                   [parse(t, [p["id"] for p in i["menu"]]) for t in texts[k * a.g:(k + 1) * a.g]]) for k, i in enumerate(batch)]
        t1 = time.time()
        rs = pool.run([(i["seed"], ids, k) for i, _, _, _, ps in groups for ids, k, _ in ps])
        R = np.array([reward(r, p[2]) for r, p in zip(rs, [p for g in groups for p in g[4]])]).reshape(a.prompts, a.g)
        A = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)
        t2 = time.time()
        opt.zero_grad()
        ntok = sum(float(g[3].sum()) for g in groups)
        for (i, ids, c, mask, _), adv in zip(groups, A):
            if np.abs(adv).max() > 0:
                (-(torch.tensor(adv, dtype=torch.float32)[:, None] * logp(m, ids, c, a.temp) * mask).sum() / ntok).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        t3 = time.time()
        log["steps"].append(dict(step=step, reward=float(R.mean()), best=float(R.max(1).mean()), std=float(R.std(1).mean()),
                                 frac=float(np.mean([r["frac"] if r else 0 for r in rs])), valid=float(np.mean([bool(p[0]) and p[2] == 0 for g in groups for p in g[4]])),
                                 tgen=t1 - t0, tsim=t2 - t1, ttrain=t3 - t2, sample=groups[0][4][0][:2]))
        print(step, {k: round(v, 3) for k, v in log["steps"][-1].items() if isinstance(v, float)}, flush=True)
        (OUT / "curve.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
