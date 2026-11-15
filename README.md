# Espoo 2040

An agent plans the roads of Espoo, Finland, for 15 years. A small learned policy beats it in every tested scenario (a scenario is the city with one seed).

Viewer: https://espoo.evinu.de. Paper: `paper/main.pdf`.

## Run

Needs Docker, Harbor 0.25 and `OPENAI_API_KEY`.

```
SEED=0 ./eval.sh
```

It runs the reference, three agent runs (`MODEL`), do nothing and an early stop.

## Task

Each year the agent starts up to two of 22 road projects and picks maintenance 0, 1 or 2, with 40 MEUR income. Two or three of ten areas grow fast; which ones shows only in `get_state`. Four MCP tools: `get_state`, `list_projects`, `forecast` (4 per year, 3 years ahead), `commit_year`. Score: total delay over 20 years, do nothing 0, best known plan 1. Rules: `task/instruction.md`, `sim/`.

`data/catalogue.json` is hand-made; its old generator could not rebuild it and was removed (`f45ca44`). Twelve projects follow real Espoo plans; only one has a source (Kehä II and Turunväylä junction, espoo.oncloudos.com/kokous/20231048-15.PDF). Costs come from road length. The ten access upgrades are made up.

## Results

One agent run per seed, scored on Intel (`evidence/failures.json`):

| Seed | 0 | 1 | 2 | 3 | 4 | Mean |
|---|---|---|---|---|---|---|
| Reference | 1.00 | 1.00 | 0.91 | 0.99 | 0.98 | 0.97 |
| gpt-6.1-sol | 0.52 | 0.60 | 0.87 | 0.93 | 0.92 | 0.77 |

On seeds 2 to 4 the lead is small (0.03 to 0.06); five runs is few.

Seeds 0 to 15: learned policy 0.97, rule planner 0.84, Gemma 3 270M after GRPO 0.83 (from 0.19, but one plan for all scenarios), reactive rule 0.56, random 0.15.

The reference (`rl/policy.py`, `rl/policy.npz`) has 83k parameters and uses no forecasts. It learned from expert plans, then PPO; the checkpoint was chosen on seeds 0 to 15. On 32 unseen scenarios it saves 34.3% of the do-nothing delay (rule planner 31.1%). Seeds without norms (16 and up) report this saved share, not the score, so the scale differs.

## Why the agent fails

By year 2 it names all 15 growing areas, yet never upgrades 11. Its plans barely change between scenarios (overlap 0.72, reference 0.44). It builds two cheap links first, and that order of spending blocks later upgrades. Adding one access upgrade closes only 1.8% of the gap, so the gap is the whole plan.

## Why the reward is hard to cheat

- A separate verifier replays the plan and scores it (`task/tests/test.sh`). It logs `replay_off`, the gap to the live run, outside the reward.
- Do nothing scores 0.
- After a stop, the remaining years get no new projects.
- Rejected actions cost nothing and do nothing.
- Scores are clipped to 0 to 1.
- Replays are bit-identical on one CPU type; AMD and Intel scores differ by at most 0.008.

## Layout

`sim` simulator, `task` Harbor task and verifier, `rl` reference policy, `grpo` Gemma training, `data` network and demand, `viz` viewer, `paper` paper.
