import argparse
import json

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import OUT, Pool, parse, prompt, reward

BASE, EVAL = "unsloth/gemma-3-270m-it", list(range(16))


def load(ckpt=None):
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    m.generation_config.max_length = None
    if ckpt:
        return tok, PeftModel.from_pretrained(m, ckpt)
    return tok, get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))


def enc(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]


def sample(m, tok, texts, n, greedy=False):
    m.eval()
    x = tok.apply_chat_template([[{"role": "user", "content": t}] for t in texts for _ in range(n)], add_generation_prompt=True, padding=True,
                                return_tensors="pt", return_dict=True)
    with torch.no_grad():
        c = m.generate(**x, max_new_tokens=64, do_sample=not greedy, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)[:, x["input_ids"].shape[1]:]
    end = torch.isin(c, torch.tensor(m.generation_config.eos_token_id))
    return c, (end.cumsum(1) - end.int()) == 0, [tok.decode(y, skip_special_tokens=True) for y in c]


def logp(m, ids, c):
    m.train()
    lg = m(input_ids=torch.cat([ids.repeat(len(c), 1), c], 1), logits_to_keep=c.shape[1] + 1).logits[:, :-1].float()
    return torch.log_softmax(lg, -1).gather(-1, c[..., None])[..., 0]


def evaluate(m, tok, pool, infos, n=4, off=False):
    with m.disable_adapter() if off else torch.no_grad():
        ps = [prompt(i) for i in infos]
        g, r = sample(m, tok, ps, 1, True)[2], sample(m, tok, ps, n)[2]
    jobs = [[(i["seed"], *parse(t, [p["id"] for p in i["menu"]])) for t in [g[k]] + r[k * n:(k + 1) * n]] for k, i in enumerate(infos)]
    rs = [pool.run([j[:3] for j in js]) for js in jobs]
    out = [dict(seed=i["seed"], text=g[k], greedy=rs[k][0], score=(rs[k][0] or {}).get("score") or 0.0,
                sampled=float(np.mean([(r or {}).get("score") or 0.0 for r in rs[k][1:]])),
                valid=float(np.mean([bool(j[1]) and j[3] == 0 for j in jobs[k]]))) for k, i in enumerate(infos)]
    f = lambda k: float(np.mean([o[k] for o in out]))
    return dict(score=f("score"), sampled=f("sampled"), valid=f("valid"), saving=float(np.mean([(o["greedy"] or {}).get("saving", 0) for o in out])), seeds=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=120)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng, pool = np.random.default_rng(a.seed), Pool(a.workers)
    tok, m = load()
    train, ev = pool.infos(range(1000, 1400)), pool.infos(EVAL)
    log = dict(args=vars(a), steps=[], evals=[dict(step=-1, name="zero-shot", **evaluate(m, tok, pool, ev, off=True))])
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    for s in range(a.warm):
        opt.zero_grad()
        for i in rng.choice(train, 4):
            t = f"plan: {', '.join(rng.permutation([p['id'] for p in i['menu']])[:rng.integers(2, 7)])}; maintain: {rng.integers(3)}"
            (-logp(m, enc(tok, prompt(i)), tok(t + "<end_of_turn>", add_special_tokens=False, return_tensors="pt")["input_ids"]).mean() / 4).backward()
        opt.step()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    for step in range(a.steps):
        if step % 10 == 0:
            e = evaluate(m, tok, pool, ev)
            log["evals"].append(dict(step=step, name="grpo" if step else "warm", **e))
            m.save_pretrained(OUT / f"ckpt-{step}")
            print("eval", step, round(e["score"], 3), round(e["sampled"], 3), round(e["valid"], 3), flush=True)
        batch = rng.choice(train, a.prompts, replace=False)
        c, mask, texts = sample(m, tok, [prompt(i) for i in batch], a.g)
        ps = [parse(t, [p["id"] for p in i["menu"]]) for i, t in zip(np.repeat(batch, a.g), texts)]
        rs = pool.run([(i["seed"], p[0], p[1]) for i, p in zip(np.repeat(batch, a.g), ps)])
        R = np.array([reward(r, p[2]) for r, p in zip(rs, ps)]).reshape(a.prompts, a.g)
        A = torch.tensor((R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4), dtype=torch.float32)
        opt.zero_grad()
        for k, i in enumerate(batch):
            g = slice(k * a.g, (k + 1) * a.g)
            if A[k].abs().max() > 0:
                (-(A[k, :, None] * logp(m, enc(tok, prompt(i)), c[g]) * mask[g]).sum() / mask.sum()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        log["steps"].append(dict(step=step, reward=float(R.mean()), best=float(R.max(1).mean()), std=float(R.std(1).mean()),
                                 frac=float(np.mean([r["frac"] if r else 0 for r in rs])), valid=float(np.mean([bool(p[0]) and p[2] == 0 for p in ps])), sample=ps[0][:2]))
        print(step, {k: round(v, 3) for k, v in log["steps"][-1].items() if isinstance(v, float)}, flush=True)
        (OUT / "curve.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
