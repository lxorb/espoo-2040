# Transport plan for Espoo

You are the transport planner of Espoo, Finland. Every year you decide which road projects to start and how much to spend on road maintenance. The city grows and drivers change routes. Your goal is to keep the time that travellers lose in traffic as low as possible.

## Model

- The road network covers Espoo and Kauniainen. Each year the morning peak hour is assigned to the network: car and truck trips take their fastest routes. Congested roads get slower, so traffic reroutes around everything you build. The share of trips made by car is fixed.
- The yearly loss is the delay of all people in cars and trucks: their travel time minus the free-flow time on roads in perfect condition, in person-hours per year.
- Population and jobs grow every year. In each city a few of the ten development areas are boom areas (`rules.boom_areas` gives the lowest and highest number). A boom area grows at a rate drawn once inside the range given in `/data/zones.json` (`growth.areas`); every other zone grows at the base rate (`growth.pop`, `growth.jobs`). Which areas boom is not stated. It shows year by year in `areas` of `get_state` (now against the base year).
- Road condition (0 to 1) falls every year, faster under heavy traffic. Worse condition lowers speed. The maintenance level sets a yearly maintenance budget (`rules.maintain.budget_meur`), spent on resurfacing the worst segments below `rules.maintain.fixes_below_cond`. A segment that falls below `rules.emergency_rebuild.below_cond` without being resurfaced gets an emergency rebuild that year, at a multiple of its resurfacing cost and with reduced capacity.

## Money and rules

- `get_state` reports the income, opex, committed instalments and the exact rules (`rules`): the number of planning years and runout years, builds per year, maintenance levels and their budgets, emergency rebuilds, forecasts per year and the longest forecast.
- Each year you may start up to `rules.builds_per_year` projects from the menu and choose one maintenance level (default `rules.maintain.default`). Maintenance and emergency rebuilds are paid from your cash. Unspent cash carries over.
- A project's cost is paid in equal instalments over its build years, the first one when you start it. A start is accepted only if your cash stays non-negative in every instalment year. During construction a project reduces capacity on the roads it works on; it opens after its build years and then costs its opex every year.
- `startable` in `get_state` lists the projects you can start this year. An action that does not fit is rejected with a reason and costs nothing.

## Tools

The MCP server `espoo` offers four tools.

- `get_state()`: year, cash, losses, network condition, development areas, projects under construction and built, startable projects, forecasts left, rules, the most delayed roads.
- `list_projects()`: the menu with cost, build years, opex, status, `jam` (the share of last year's network delay on the roads the project changes) and those roads.
- `forecast(plan, years)`: simulates a plan from the current state without changing anything. `plan` maps a year offset (`"0"` is this year) to that year's actions. Its growth for each area blends the middle of the area's range with the rate the area has shown so far, trusting the shown rate by `rules.forecast_trust`. A forecast covers at most `rules.forecast_years` years, and you have `rules.forecasts_per_year` forecasts per year; the count resets when you commit a year.
- `commit_year(actions)`: carries out this year's actions and advances one year.

```json
[{"build": "olari-link"}, {"build": "kera-access"}, {"maintain": 2}]
```

Static copies of the network, the zones and the full catalogue are in `/data` (`net.json`, `zones.json`, `catalogue.json`) for your own analysis.

## Score

The score is the total loss over the planning years plus the runout years, in which no new actions are taken, maintenance is at level `rules.maintain.runout_level`, your projects keep working and the city keeps growing. Commit one year at a time. When you stop, the remaining planning years run with no new projects and the default maintenance, so stopping early leaves the city to itself.

The reward is the loss you saved compared with doing nothing, divided by the saving of the best known plan for this city, clipped to the range 0 to 1. Doing nothing scores 0.
