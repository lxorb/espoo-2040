# Espoo 2040

An agent plans the roads of Espoo, Finland, for 15 years. A small learned policy beats the frontier agent on all 5 agent seeds (0.97 vs 0.77) and holds up on 32 unseen scenarios (a scenario is the city with one seed).

Viewer: https://espoo.evinu.de. Paper: `paper/main.pdf`.

## Judge quickstart

Needs Docker and Harbor 0.25.
- `SEED=0 harbor run -y -p task -a oracle`: reward about 1.0, 2 minutes on first build.
- `SEED=0 harbor run -y -p task -a nop`: reward 0.0, 1 minute.
- `SEED=0 harbor run -y -p task -a opencode -m openai/gpt-6.1-sol`: needs `OPENAI_API_KEY`.
- `SEED=0 MODEL=provider/model ./eval.sh`: reference, three agent runs, do nothing, early stop.
- Tests: `uv run --with mcp==1.30.0 pytest -q tests` (CI is defined in `.github/workflows/ci.yml`).

## Task

Each year the agent starts up to two of 22 road projects and picks maintenance 0, 1 or 2 (income 40 MEUR). Two or three of ten areas grow fast, visible only in `get_state`. MCP tools: `get_state`, `list_projects`, `forecast` (4 per year, 3 years ahead), `commit_year`. Score: total delay over 20 years; do nothing 0, best known plan 1. Rules: `task/instruction.md`.

`data/catalogue.json` is hand-made: twelve projects follow real Espoo plans, the ten access upgrades are made up.

## Results

One agent run per seed, Intel scores (`evidence/failures.json`):

| Seed | 0 | 1 | 2 | 3 | 4 | Mean |
|---|---|---|---|---|---|---|
| Reference | 1.00 | 1.00 | 0.91 | 0.99 | 0.98 | 0.97 |
| gpt-6.1-sol | 0.52 | 0.60 | 0.87 | 0.93 | 0.92 | 0.77 |

Seeds 2 to 4 are close (0.03 to 0.06); five runs are few.

Seeds 0 to 15: learned policy 0.97, rule planner 0.84, Gemma 3 270M after GRPO 0.83 (from 0.19; two of three runs end on one fixed plan, one reacts to one area), reactive rule 0.56, random 0.15.

The reference (`rl/policy.npz`, 83k parameters, no forecasts) learned from expert plans, then PPO; checkpoint picked on seeds 0 to 15. On 32 unseen scenarios (seeds 16 to 47, stricter norms) it scores 0.96 and saves 34.3% of the do-nothing delay (rule planner 31.1%).

Under stricter cash rules (`tools/v2.py`: emergency rebuilds only when cash allows), the reference, not retrained, falls below the rule planner (0.68 against 0.85); with maintenance 2 it scores 0.87.

## Why the agent fails

Its plan barely varies (overlap 0.72; reference 0.44). It always opens with two cheap links and maintenance 2; the reference opens with `espoonvayla-2x2` and maintenance 1. The budget allows about three starts: the reference uses one widening and two access upgrades, the agent four cheaper projects before any access upgrade. On seed 0 it names the growing areas in 2028 and could afford an access upgrade, but starts two widenings instead. No single fix closes the gap: replays with the reference's first move (0.69) or the right targets (0.63) score lower, because the pieces of a plan interact through the budget (`paper/figs/review.py`). The fixed plan GRPO learned (0.82 on these seeds) also beats the agent.

## Why the reward is hard to cheat

- A separate verifier replays and scores the plan (`task/tests/test.sh`); its `replay_off` log is not part of the reward.
- Do nothing scores 0.
- After a stop, no new projects start. Stopping the reference after 1 to 3 years scores 0.15, after 4 to 6 years 0.76, after 7 or more the full score (its plan is done by then).
- Rejected actions do nothing.
- Scores are clipped to [0, 1].
- Replays are bit-identical on one CPU type; AMD and Intel differ by at most 0.008.

## Known limits

- A fixed plan that ignores the scenario scores 0.83 on seeds 0 to 15 and 1.04 on seed 0: the frozen best known plans of seeds 0, 7, 9 and 11 are beatable.
- About three starts per run fit the budget; debt has no floor and is not scored.
- The agent runs predate the frozen norms (Harbor logged 0.78 on seed 0, rescored 0.52) and the task text on negative cash.
