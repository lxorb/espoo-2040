# Transport plan for Espoo, 2026 to 2040

You are the transport planner of Espoo, Finland. For 15 years you decide every year which projects to start, which roads to resurface and which signals to retime. The city grows, drivers change routes, and travellers switch between car and public transport as travel times change. Your goal is to keep the total travel delay of all travellers as low as possible.

## Model

- The road network covers Espoo and Kauniainen with about 520 segments. Each year the morning peak hour is assigned to the network: car and truck trips take their fastest routes, and each traveller chooses between car and public transport by travel time and price. Congested roads get slower, so traffic and mode share respond to everything you build.
- Delay is counted in person-hours per year: car travellers' time above free-flow travel time, plus public transport travellers' time above the free-flow car time. The peak hour is scaled to a year.
- Population and jobs grow every year. City-wide rates are fixed; each development area (Finnoo, Kera, Hista and others) has its own rate, drawn once per city inside the range given in `/data/zones.json` (`growth.areas`). You learn the drawn rates only as they show up year by year in `get_state`.
- Road condition (0 to 1) falls every year, faster with heavy traffic and trucks. Worse condition lowers speed. A segment whose condition falls below 0.2 gets an emergency rebuild that year at six times the resurfacing cost and with half capacity.

## Money

- Income is 120 MEUR per year. Unspent cash carries over.
- A project's cost is paid in equal instalments over its build years, the first one when you start it. A start is accepted only if your cash stays non-negative in every instalment year, given income, earlier instalments and opex. Projects open after their build years and then cost their opex every year. During construction some projects reduce capacity on the roads they work on.
- Resurfacing a segment costs its listed price, restores its condition to 1 and cuts its capacity to 70% for that year.
- Retiming the signals of a segment costs 0.02 MEUR and adds 8% capacity, which fades by 30% per year.
- An action that does not fit the budget or the rules is rejected with a reason and costs nothing.

## Tools

The MCP server `espoo` offers five tools.

- `get_state()`: year, cash, committed instalments, last year's delay and car share, population and jobs per development area with growth so far, projects under construction and built, forecasts left, the 25 most congested and the 25 worst-condition segments.
- `list_projects()`: the catalogue: road links, extra lanes, junctions, bus lanes, rail, bus lines, tolls, parking prices and park-and-ride, with cost, build years, opex, price levels, status and the roads they change.
- `get_segments(sort, limit, ids)`: the segment table with flow, volume/capacity, delay, condition and resurfacing cost.
- `forecast(plan, years)`: simulates a plan from the current state without changing anything. `plan` maps a year offset (`"0"` is this year) to a list of actions. It uses the expected growth of every area (the middle of its range), not the drawn one. You have 6 forecasts per year; the count resets when you commit a year.
- `commit_year(actions)`: carries out this year's actions and advances one year.

Actions:

```json
{"build": "keha2-north"}
{"build": "toll-lansivayla", "level": 1}
{"resurface": 124}
{"retime": 124}
```

`level` picks the price of a toll or parking project (index into its `levels`, default 0).

Static copies of the network, the zones and the catalogue are in `/data` (`net.json`, `zones.json`, `catalogue.json`) for your own analysis.

## Score

The score is the total delay over the 15 planning years plus 5 runout years without new actions, in which your projects keep working and the city keeps growing. Commit one year at a time. When you stop, all remaining years are run with no actions, so stopping early leaves the city to decay.

The reward is the delay you saved compared with doing nothing in every year, divided by the saving of the best known plan for this city, clipped to the range 0 to 1. Doing nothing scores 0.
