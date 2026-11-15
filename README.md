# Espoo 2040

[![CI](https://github.com/lxorb/espoo-2040/actions/workflows/ci.yml/badge.svg)](https://github.com/lxorb/espoo-2040/actions/workflows/ci.yml)

An agent plans the roads of Espoo, Finland, for 15 years. A small learned policy beats it in every tested scenario (the city with one seed).

Viewer: https://espoo.evinu.de. Paper: `paper/main.pdf`.

## Judge quickstart

Needs Docker and Harbor 0.25.
- `SEED=0 harbor run -y -p task -a oracle` (or `-a nop`): reward 1.0 (0.0), 2 minutes on first build.
- `SEED=0 harbor run -y -p task -a opencode -m openai/gpt-6.1-sol`: needs `OPENAI_API_KEY`.
- `SEED=0 ./eval.sh`: reference, three serial agent runs (6 GB each; `N=3` parallel), do nothing, early stop.

## Task

Each year the agent starts up to two of 22 road projects and picks maintenance 0, 1 or 2 (income 40 MEUR). Two or three of ten areas grow fast, visible only in `get_state`. MCP tools: `get_state`, `list_projects`, `forecast` (4 per year, 3 years ahead), `commit_year`. Score: total delay over 20 years; do nothing 0, best known plan 1. Rules: `task/instruction.md`.

`data/catalogue.json` is hand-made. Twelve projects follow real Espoo plans; one has a source (espoo.oncloudos.com/kokous/20231048-15.PDF). Costs follow road length; the ten access upgrades are made up.

## Results

One agent run per seed, Intel scores (`evidence/failures.json`):

| Seed | 0 | 1 | 2 | 3 | 4 | Mean |
|---|---|---|---|---|---|---|
| Reference | 1.00 | 1.00 | 0.91 | 0.99 | 0.98 | 0.97 |
| gpt-6.1-sol | 0.52 | 0.60 | 0.87 | 0.93 | 0.92 | 0.77 |

Seeds 2 to 4 are close (0.03 to 0.06); five runs are few.

Seeds 0 to 15: learned policy 0.97, rule planner 0.84, Gemma 3 270M after GRPO 0.83 (from 0.19; two of three runs end on one fixed plan, one reacts to one area), reactive rule 0.56, random 0.15.

The reference (`rl/policy.npz`, 83k parameters, no forecasts) learned from expert plans, then PPO; checkpoint picked on seeds 0 to 15. On 32 unseen scenarios it saves 34.3% of do-nothing delay (rule planner 31.1%). Norms for 16 to 47 (`tools/norms.py`) are stricter than for 0 to 15, so not comparable. Seeds 48 and up, or non-default YEARS/RUNOUT, fall back to saving/nothing (verifier warns).

Under stricter cash rules (`tools/v2.py`: emergency rebuilds only when cash allows), the reference, not retrained, falls below the rule planner (0.68 against 0.85); with maintenance 2 it scores 0.87.

## Why the agent fails

By year 2 it names all 15 growing areas, yet never upgrades 11. Its plans barely vary (overlap 0.72; reference 0.44). Building two cheap links first blocks later upgrades. Adding one access upgrade closes only 1.8% of the gap.

## Why the reward is hard to cheat

- A separate verifier replays and scores the plan (`task/tests/test.sh`); its `replay_off` log is not part of the reward.
- Do nothing scores 0.
- After a stop, no new projects start. Stopping the reference after 3 years scores 0.15, after 5 years 0.76.
- Rejected actions do nothing.
- Scores are clipped to [0, 1].
- Replays are bit-identical per CPU type; AMD and Intel differ by at most 0.008.
