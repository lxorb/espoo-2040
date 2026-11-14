# Espoo 2040

An agent plans Espoo's road network for 15 years: each year it starts up to two road projects from a menu of 22 and sets a maintenance level, with 40 MEUR of income. Some areas of the city boom and the agent is not told which. The score is the total time lost in traffic over 20 years, scaled so that doing nothing is 0 and the best known plan is 1.

Viewer: https://espoo.evinu.de (press `p` three times for the guided tour). Paper: `paper/main.pdf`.

## Run

Needs Docker, Harbor 0.25 and `OPENAI_API_KEY`.

```
SEED=0 ./eval.sh
```

This runs the reference policy, the agent (`MODEL`, default `openai/gpt-6.1-sol`), doing nothing and an early stop on one city and prints the rewards.

## Results

| Planner | Score |
|---|---|
| Learned reference policy (83k parameters), seeds 0 to 4 through Harbor | 0.98 |
| GPT-6.1 agent, seeds 0 to 4 through Harbor | 0.77 |
| Learned reference policy, seeds 0 to 15 | 0.97 |
| Rule planner, seeds 0 to 15 | 0.84 |
| Gemma 3 270M after GRPO, seeds 0 to 15 | 0.83 |
| Reactive rule, seeds 0 to 15 | 0.56 |
| Random, seeds 0 to 15 | 0.15 |

On 32 unseen cities the reference saves 34.3% of the do-nothing delay, the rule planner 31.1%.

Scores are bit-identical on the same CPU type and differ by at most 0.01 between AMD and Intel.
