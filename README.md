# Espoo 2040

Open city traffic data breaks today's frontier models, then trains them to grow stronger.

We took public data about Espoo, Finland (roads, people, jobs, traffic counts) and built a traffic simulator from it. An agent is the city's transport planner for 15 years. Each year it picks road projects and a maintenance level, with 40 MEUR of income. Two or three areas of the city grow fast, and the agent has to notice which ones.

A frontier model (`gpt-6.1-sol`) plays this badly. A small network (83k parameters) trained in the same simulator plays it well. Gemma 3 4B, trained with GRPO for one weekend, gets better than random plans.

- Viewer: https://espoo.evinu.de
- Paper: `paper/main.pdf` (method, all numbers, limits)
- Every number in the paper comes from files in `evidence/` through `paper/figs/make.py`.

## Run it (for judges)

You need Docker and Harbor 0.25. The first build compiles the traffic model, so it can take about 5 minutes. After that a run takes 1 to 2 minutes.

```bash
SEED=0 harbor run -y -p task -a oracle
```
Expected reward: about 1.0 (our learned policy).

```bash
SEED=0 harbor run -y -p task -a nop
```
Expected reward: 0.0 (do nothing).

```bash
SEED=0 harbor run -y -p task -a opencode -m openai/gpt-6.1-sol
```
Needs `OPENAI_API_KEY`. One run costs about 0.4 USD.

```bash
SEED=0 MODEL=openai/gpt-6.1-sol ./eval.sh
```
Runs the reference, the agent (`N=3` runs them in parallel, about 6 GB each), do nothing and an early stop, then prints a table.

```bash
uv run --group dev --with mcp==1.30.0 pytest -q tests
```
Unit tests for the reward, replays and task copies.

`SEED` can be 0 to 47. Seeds 0 to 15 are for validation and 16 to 47 are the test split; training used seeds from 1000 up. For other seeds the verifier falls back to "share of delay saved" and prints a warning.

## The task

- 22 road projects (`data/catalogue.json`). Twelve follow real Espoo plans. Ten "access upgrades", one per area, are made up by us and all cost the same.
- Each year: start up to two projects, pick maintenance 0, 1 or 2, then `commit_year`.
- MCP tools: `get_state`, `list_projects`, `forecast` (4 per year, 3 years ahead), `commit_year`.
- Road works close lanes while they last, so a new road makes traffic worse at first.
- Score: total time lost in traffic over 20 years. Do nothing = 0, best plan we know = 1.
- Full rules: `task/instruction.md`.

## How the city is built

`data/` turns public data into the simulator, in six steps:
1. a road graph from OpenStreetMap;
2. zones from the HSY population and job grids;
3. a peak-hour trip matrix with a gravity model;
4. fitting to real traffic counts (we report the GEH fit in the paper);
5. the project list;
6. a yearly traffic model (static user equilibrium with BPR delay curves) with hidden growth.

The same steps work for another city if it has the same kind of open data. That is the main idea of the paper.

## Results

Validation seeds 0 to 15, one agent run per seed (`paper/figs/results.tex`):

| Method | Score | 95% CI |
|---|---|---|
| Learned policy (reference) | 0.97 | 0.95 to 0.99 |
| Rule planner | 0.84 | 0.78 to 0.89 |
| Frontier agent (`gpt-6.1-sol`) | 0.78 | 0.67 to 0.88 |
| Greedy 5-year, maintenance 2 | 0.56 | 0.47 to 0.65 |
| Random | 0.15 | 0.09 to 0.23 |

- The reference beats the agent on 15 of 16 seeds (sign test p = 0.0003).
- The agent is not bad everywhere. Its runs go from 0.32 to 0.99, and on seed 13 it beats the reference.
- Five older runs (seeds 0 to 4, older task text) gave 0.77.
- Seven runs where we told the agent to attack the reward found no exploit (mean 0.70).
- On 32 unseen test scenarios the reference saves 34.3% of the do-nothing delay, and the rule planner 31.1%.

GRPO on Gemma 3 4B (details in the paper, section 7):
- The model writes a short `<think>` block and then a plan. Project names are random labels in every scenario, so it cannot memorise a list. The reward looks only at the result.
- Two training seeds, about one GPU hour each. On 32 unseen test seeds it saves 18.0% of the delay, against 10.4% before training and 9.1% for random plans.
- It beats random plans on 25 and 27 of 32 seeds (p = 0.001 and p < 0.0001). We wrote this test down before we saw the results.
- It does not do budget math yet, and a good fixed plan still does better (28.2%). We expect larger models to close this gap.

## Why the agent fails

- Its plans look almost the same in every scenario (overlap 0.65; reference 0.50).
- It names the growing areas by year 3, but leaves most of them without an upgrade.
- The budget allows about three big starts. The agent often spends them on cheap links and widenings first, and then the money is gone.
- It trusts the 3-year forecast. A 120 MEUR upgrade looks bad in 3 years (two years of road works) and good over 15. Our learned policy gets worse too if we make it check that forecast (0.97 drops to 0.40).

## Why the reward is hard to cheat

- A separate container replays the plan and scores it (`task/tests/test.sh`). The agent cannot write the score.
- Do nothing scores 0. Stopping early is allowed but costs: stopping the reference after 3 years gives 0.15, after 5 years 0.76.
- Rejected actions do nothing. Scores are clipped to [0, 1].
- `task/tests/exploits.py` and the paper's exploit table list what we tried.

## Known limits

We prefer to say these ourselves:
- One agent run per seed and one model. That is few runs.
- "Best known plan" is not a proven best. On seeds 0, 7, 9 and 11 a fixed plan scores above 1 before clipping. New norms are ready (`evidence/norms_v1.json`) but not adopted yet, because they move every number.
- Debt has no floor and emergency repairs are always paid. With stricter cash rules (`tools/v2.py`) the reference, not retrained, falls below the rule planner. Retrained under those rules, it leads again by a small margin.
- Seed 4 depends a lot on one short single-lane road piece (Kalevalantie). This is a known weakness of static traffic models.
- Replays are bit-identical on the same CPU type. AMD and Intel differ by at most 0.008.
- The traffic model has one peak hour, no queues, and a fixed car share.

## Repo map

| Folder | What is inside |
|---|---|
| `task/` | the Harbor task: instruction, MCP server, verifier, oracle |
| `sim/` | traffic model, scoring, norms |
| `data/` | the data pipeline and its outputs |
| `rl/` | the learned policy (BC then PPO) |
| `grpo/` | GRPO for Gemma (`grpo/reason.py` for the 4B runs) |
| `tools/` | baselines, exploits, norms, stricter rule variants |
| `evidence/` | raw results of every run |
| `paper/` | paper source; `paper/build.sh` rebuilds it |
| `viz/` | the viewer |
