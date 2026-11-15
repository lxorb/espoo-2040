# Experiments

Every number in the paper is a macro in `paper/figs/numbers.tex` or a row in a `paper/figs/*.tex` table. All of them are written by `paper/figs/make.py`, run through `paper/build.sh`. This file says where each one comes from.

Score: `(D_nothing - D) / (D_nothing - D_best)` with `sim/norms.json` (frozen at commit `d24719a`), clipped to [0, 1] unless a row says raw. Saved share: `1 - D / D_nothing` on the same seed. Delay `D`: total person-hours over 2026 to 2045.

## Rebuild

`paper/build.sh` needs only committed files: `sim/norms.json`, `data/catalogue.json`, `evidence/`, and the summaries in `paper/figs/data/`:

| File | Content | Made by |
|---|---|---|
| `runs.json` | per-seed delays, spend, builds, maintenance of all simulated and Harbor runs | `collect.py` from scratch run folders and `jobs/` |
| `rl.json` | RL strategy rows (up to `CUT` = 5280 episodes for PPO from scratch), PPO and Gumbel curves, search, `jam` ablation, unseen-seed results | `collect.py` from `runs/rl/` |
| `grpo.json` | GRPO evaluation per step (greedy plan and saving per seed) and run A strategy rows | `collect.py` from `runs/grpo/` |
| `growth.json`, `booms.json` | seed 1 area growth; growing areas of seeds 0 to 15 | `prep.py` |

When `runs/` and `jobs/` exist, `build.sh` first runs `collect.py` to refresh these summaries. Tested on a fresh `git clone` (with the pending changes to `paper/figs/` and `paper/build.sh` copied in): the build gives the same `numbers.tex`, tables and figures as the working copy, and 10 pages.

## Data split

| Split | Seeds | Use |
|---|---|---|
| train | 1000+ (BC, GRPO), 2000+ (PPO) | training only |
| validation | 0 to 15 | checkpoint choice, all headline scores |
| test | 16 to 47 | never used for training or selection |

The learned policy checkpoint was chosen on validation seeds 0 to 15 (highest mean plus worst score). Every validation number for it is therefore optimistic; the test split measures by how much (`hoGap`).

## Versions and compute

| Item | Value | Source |
|---|---|---|
| Harbor | 0.25.0 | `jobs/gate4-s*/lock.json` |
| OpenCode | 1.19.5 | `jobs/gate4-s*/*/agent/trajectory.json` (`agent.version`) |
| Agent model | `openai/gpt-6.1-sol` | `jobs/gate4-s*/config.json` |
| Agent runs | 2026-11-14 17:14 to 17:48 UTC (gate4, seeds 0 to 4) | `jobs/gate4-s*/result.json` |
| Harbor reference runs | 2026-11-14 18:53 to 18:58 UTC (ref3), 17:24 (refrl, old weights) | `jobs/ref3-s*/result.json`, `jobs/refrl-s*/result.json` |
| Training VMs | espoo-vm (swedencentral), espoo-vm2 and espoo-vm3 (northeurope): Azure D32s, Intel Xeon Platinum 8573C | Azure; not Verda |
| Paper runs | local AMD machine (see conflicts) | `paper/figs/data/runs.json` |
| BC, PPO, GRPO CPU and wall time | pending, from the rl and grpo owners | |

## Numbers

Date for all rows below: 2026-11-14, unless noted. "scratch" means the producing script is not in the repository (see conflicts).

| Macro or table | Meaning | Script | Command | Seeds |
|---|---|---|---|---|
| `table.tex` rows Do nothing to Best known; `rlScore`, `rlcheckScore`, `reactiveMOne`, `reactiveMTwo`, `greedyFiveMOne`, `greedyFiveMTwo` | score, delay, saved share, 2045 delay, spend per method; maintenance level picked per baseline by the higher mean | baselines: `rl/eval.py` `simple()` (then version) with maintenance forced by scratch `bl2.py`; learned policy, forecast check, rule planner: scratch `drive.py` (calls `task/environment/server/server.py` tools) | `python -m rl.eval --policies nothing,random,reactive,greedy1,greedy5`; `python bl2.py planner,reactive,random,greedy1,greedy5`; `python drive.py <seed> rl|rlcheck|plan` | 0 to 15 (table columns: 0 to 4) |
| `agentScoreMean`, `agentN`, `agentSeeds`, `agentLostC`, `agentUsdMin/Max`, `agentMinMin/Max`, `agentGetStateMin/Max`, `agentForecastMin/Max`, `nBooms`, `agentBoomHits`, `agentBoomEarly`, `gapRlA`, `agent.tex` | frontier agent runs: score, delays, cost, minutes, tool calls, access upgrades in growing areas | Harbor, `paper/figs/collect.py` | `harbor run -p task -a opencode -m openai/gpt-6.1-sol` (job names gate4-s0 to s4, `SEED` env) | 0 to 4 |
| `refrlN`, `refrlGap` | largest relative delay gap between Harbor reference (ref3) and our own run of the same weights | Harbor oracle `REF=rl`, `collect.py` | `harbor run -p task -a oracle` with `REF=rl` | 0 to 4 |
| `refoldLostC` | older policy (BC only, always m=2) on seed 2 via Harbor | Harbor (refrl) | as above, old weights | 2 |
| `rlLostC`, `rlBoomEarly`, `rlBoomAny`, `rlN` | learned policy delay on seed 2; cities with a growing-area access upgrade by 2029 or at all | scratch `drive.py` | as above | 0 to 15 |
| `bestPctMin`, `bestPctMax` | share of D_nothing saved by D_best | `sim/norms.json` | | 0 to 15 |
| `nothingAbsMax`, `nothingRatioMin`, `nothingRatioMax` | largest raw do-nothing score; growth of yearly delay 2026 to 2045 under do nothing | `rl/eval.py` nothing runs | as table | 0 to 15 |
| `rlsearchScore`, `rlsearchWorse` | learned policy plus Gumbel search (16 sims, depth 6) | `rl/eval.py` (search) | file `runs/rl/search-eval.json` | 0 to 15 |
| `ablation.tex`, `jamFew`, `jamFewGap`, `jamFewMinWith`, `jamFewMinWithout`, `jamMany`, `jamManyGap` | BC with and without the `jam` feature | rl owner | file `runs/rl/ablation-jam.json` | 0 to 15 |
| `heldout.tex`, `hoN`, `hoSeeds`, `hoRl`, `hoRlEval`, `hoGap`, `hoWinPlanner`, `hoWinReactive` | saved share on unseen seeds and on validation seeds; wins of the learned policy | rl owner | file `runs/rl/heldout.json` | 16 to 47 and 0 to 15 |
| `curves.pdf` (a) | mean score of PPO and Gumbel trained from scratch | `rl/ppo.py`, Gumbel path of old `rl/train.py` (removed in `7e15d8d`) | files `runs/rl/vm/6fae939a/runs/rl/ppocold2-*.json`, `gz2-*.json` | 0 to 15 |
| `evo*` macros, `evolution.pdf` (b to e) | plan statistics per RL checkpoint | `rl/strategy.py` | file `runs/rl/strategy.json` | 0 to 15 |
| `grpo*` macros, `curves.pdf` (b), `evolution.pdf` (a) | GRPO scores and plan statistics per step | `grpo/train.py`, `grpo/strategy.py` | files `runs/grpo/curve.json`, `runs/grpo/strategy.json` (run A) | 0 to 15 |
| `cats.tex` | project types, cost and build-year ranges | `make.py` | from `data/catalogue.json` | |
| `problem.pdf` | growth of seed 1, project menu, do-nothing delay | `prep.py`, `make.py` | `uv run python paper/figs/prep.py` | 1 and 0 to 2 |
| `failures.tex`, `fail*` macros | failure taxonomy of the frontier agent against the reference on the Harbor plans: plan overlap (Jaccard), year the agent first names the growing areas, growing areas never upgraded, upgrade lag, years an upgrade was not affordable, maintenance and rebuild spend shares, forecasts and tool calls | `paper/figs/failures.py` | `uv run python paper/figs/failures.py` (espoo-vm, Intel Xeon 8573C, 2026-11-15) | 0 to 4 |
| `counterfactual.tex`, `cf*` macros, `gapIntelMean`, `gapIntelLo`, `gapIntelHi`, `gapIntelWins`, `gapIntelSignP` | replays of the agent plan with one change (early access upgrade of the fastest area, maintenance 1, both) and of the reference with maintenance 2; score gap reference minus agent on Intel with 95% CI and sign test | `paper/figs/counterfactual.py`, `paper/figs/failures.py` | `uv run python paper/figs/counterfactual.py` (espoo-vm, Intel, 2026-11-15) | 0 to 4 |
| `exploits.tex`, `Exploit*` macros | exploit strategies against the MCP server and verifier: mean and best reward per strategy, guard that stops it; reference reward with 95% CI; year the reference plan is complete; first year of negative cash (reference, do nothing); stop after k years; reference with maintenance 0 and 2 (paired wins, one-sided sign test); end cash; blind and growth-aware access plans; malformed inputs accepted | `task/tests/exploits.py` | `WORKERS=8 uv run --with mcp==1.30.0 --with httpx python task/tests/exploits.py` (espoo-vm3, Intel Xeon 8573C, 2026-11-15) | 0 to 15 |
| `test.tex`, `validation.tex`, `test*`/`validation*` macros | saved share with 95% bootstrap CI, worst seed, paired tests against the learned policy | `rl/eval.py`, `make.py` | `python -m rl.eval --seeds 0-47` writes `evidence/test.json` | 0 to 47 |

Statistics helpers in `make.py`: `boot` (mean and 95% bootstrap CI, 10,000 resamples of seeds, RNG seed 0), `paired` (reference wins, one-sided sign test, one-sided Wilcoxon signed-rank on non-zero differences).

## Conflicts with README or paper

1. Most table numbers were computed on the local AMD machine, not on the Intel VMs. Across CPU types the score can differ by up to 0.008. They should be replaced by `rl/eval.py` output from an Intel VM (`evidence/test.json`).
2. Learned policy, forecast check and rule planner numbers come from scratch `drive.py`, and maintenance-forced baselines from scratch `bl2.py`. Neither script is in the repository. `rl/eval.py` now covers the same policies (`deploy`, `planner`, `reactive:2`, ...) but not `rlcheck`.
3. PPO from scratch kept training after the paper text was written. `collect.py` freezes its strategy rows and its curve at 5280 episodes (`CUT`), which gives the values the text was written with (`evoColdEps` 5280, `evoColdScore` 0.71). The live file now reaches 20640 episodes with score 0.65; commit fff88a6 holds those live values in `numbers.tex`.
4. `curves.pdf` (a) and the Gumbel numbers come from stored runs; the Gumbel training path is no longer in the code.
5. The raw run files (`runs/`, `jobs/`, `viz/data/`) are gitignored. Their small summaries are committed in `paper/figs/data/` (see Rebuild), so the figures rebuild from a fresh clone, but the summaries themselves can only be regenerated on a machine that has the raw files.
6. README said the agent lacks cash for a 120 MEUR upgrade because it buys cheap links first and keeps maintenance at level 2 ("the reference plan with maintenance 2 scores 0.52 on seed 0"). The replays do not support this: the agent with maintenance 1 loses on all five seeds (`cfMaintOneDiff`), a growing-area upgrade is unaffordable in about as many years for the agent as for the reference (`failBlockedAgent`, `failBlockedRef`), and level 2 is only `failMaintTwoShare` of the agent's spend. Forcing maintenance 2 on the reference does cost it `cfRefMaintTwoDiff`, but the effect is not symmetric. Adding one access upgrade closes only `cfAccessGapShare` of the gap, so the gap comes from the whole plan, not one choice. README is corrected.
7. The README score table used AMD numbers. On Intel the reference minus agent gaps on seeds 0 to 4 are 0.482, 0.397, 0.032, 0.056, 0.058 (mean `gapIntelMean`, 95% CI [`gapIntelLo`, `gapIntelHi`], reference wins `gapIntelWins`, sign p `gapIntelSignP`); README now shows the Intel scores from `evidence/failures.json`.
8. Exploit suite on all 16 seeds (`exploits.tex`). The reference plan is complete after year `ExploitPlanDoneYear` and its cash first turns negative in year `ExploitFirstNegative` on every seed; after that every new action is rejected, so stopping after 7 or more years gives exactly the full score (`ExploitStopSevenGap`). Debt has no floor because emergency rebuilds are always paid (mean end cash `ExploitRefDebt` MEUR; do nothing reaches debt in year `ExploitNothingFirstNegative`, end cash `ExploitNothingDebt`). The reference with maintenance 0 beats it on `ExploitMaintZeroWins` of `ExploitMaintZeroN` seeds (sign p `ExploitMaintZeroP`) but is lower on average (`ExploitMaintZeroGap`) because of a few large losses; the reference always chooses maintenance 1. No malformed input was accepted (`ExploitAcceptedBad`); the JSON does not store the number of malformed inputs tried (28 in the script). Whether the rules change is the owner's decision; README and paper conclusions are unchanged.
