# Espoo 2040

An agent plans the roads of Espoo, Finland, for 15 years. A small learned policy beats it on every tested city.

Viewer: https://espoo.evinu.de. Paper: `paper/main.pdf`.

## Run

Needs Docker, Harbor 0.25 and `OPENAI_API_KEY`.

```
SEED=0 ./eval.sh
```

It runs the reference, three agent runs (`MODEL`, default `openai/gpt-6.1-sol`), do nothing and an early stop.

## Task

Each year the agent starts up to two of 22 road projects and picks maintenance level 0, 1 or 2. Income is 40 MEUR a year. Two or three of ten areas grow fast; which ones shows only in `get_state`. Four MCP tools: `get_state`, `list_projects`, `forecast` (4 per year, 3 years ahead), `commit_year`. The score is total delay over 20 years: do nothing is 0, the best known plan is 1. Rules: `task/instruction.md`, `sim/`.

## Results

One agent run per seed (`jobs/ref3-s*`, `jobs/gate4-s*`):

| Seed | 0 | 1 | 2 | 3 | 4 | Mean |
|---|---|---|---|---|---|---|
| Reference | 1.00 | 1.00 | 0.91 | 0.99 | 0.98 | 0.98 |
| gpt-6.1-sol | 0.52 | 0.60 | 0.88 | 0.93 | 0.92 | 0.77 |

On seeds 2 to 4 the lead is small (0.03 to 0.06). Five runs is a small sample. The agent jobs ran before `sim/norms.json` was frozen; the table rescores their delays with it.

Seeds 0 to 15: learned policy 0.97, rule planner 0.84, Gemma after GRPO 0.83, reactive rule 0.56, random 0.15.

The reference (`rl/policy.py`, `rl/policy.npz`) has 83k parameters and uses no forecasts. It learned from expert plans, then PPO. We chose its checkpoint on seeds 0 to 15. On 32 unseen cities it saves 34.3% of the do-nothing delay (rule planner 31.1%), 3.6 points less than on seeds 0 to 15.

## Why the agent fails

It makes nearly the same plan in every city. It calls `get_state` only 7 or 8 times, so it rarely sees which areas grow. A second access upgrade comes in 2034 or never. Its 3-year forecasts show the cost of road works, not the later gain. It fails where growth needs an early answer (seeds 0 and 1).

## Why the reward is hard to cheat

- A verifier in a separate container replays the plan (`task/tests/test.sh`, `replay_off` must be 0).
- Do nothing scores 0.
- After a stop, the remaining years get no new projects and default maintenance. The reference's first 3 years alone score 0.05 on seed 0 (0.05 to 0.38 on seeds 0 to 4).
- Rejected actions cost nothing and do nothing.
- Scores are clipped to 0 to 1.
- Replays are bit-identical on one CPU type; AMD and Intel differ by at most 0.008.

## Smaller model

GRPO lifts Gemma 3 270M from 0.19 to 0.83. But it learns one plan for all cities and ignores growth.

## Layout

- `sim` simulator
- `task` Harbor task and verifier
- `rl` reference policy
- `grpo` Gemma training
- `data` network and demand
- `viz` viewer
- `paper` paper and figures
