# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/denisyky/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/denisyky/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hourly Twitch CCU forecaster for GTA VI launch day: log-space quantiles fitted from rolling-midnight launch comparables. Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, $250 in USDC.

## Model

`main.ts` (Node 22, standard library only, offline, deterministic) reads one case `/input/<id>.in` and writes `/output/<id>.out` with five quantiles (5%, 25%, 50%, 75%, 95%) of Twitch viewers in the GTA VI category for each hour 2–24 of release day (UTC).

For each hour it works in log space with the ratio `viewers(h) / viewers(1)` of past big launches, fitted at runtime from the case's own `comparables`:

1. **Pool: rolling-midnight launches only.** Comparables whose unlock was a rolling midnight-local console launch and whose research tier is "comparable" (excludes early-access-contaminated / weak launches and the GTA VI category's own pre-release history). Rolling-midnight games share GTA VI's unlock shape (NZ → Europe → UK → US East → US West); single-global-unlock games spike at one hour and would wreck the quantiles. Fallbacks relax the pool (rolling any tier, then all) if a tier is too small.
2. **Median: log(p1) regression.** Per hour, an OLS fit of `log r_h` on `log(p1)`, blended 85/15 with the pool median of `log r_h`. Big launches grow less from hour 1 — across the rolling pool, corr(log r_20, log p1) ≈ −0.76 — so conditioning on the observed hour-1 level cuts the typical miss from ~60% (pooled median) to ~44% in leave-one-out tests over the comparables.
3. **Spread: residual SD.** Per-hour residual SD of that fit, floored at 0.35 and capped at 1.0, 3-point smoothed across hours; quantiles are log-normal, `q_τ = p1 · exp(μ_h + z_τ · sd_h)`. The median curve is smoothed the same way.

Validation (leave-one-out over the big rolling-midnight comparables in the kit): pinball loss ≈ 0.25, typical miss ≈ 44%, 90% coverage ≈ 0.86 — vs ≈ 0.35 / 70% for the house baseline (flat persistence, practice loss ≈ 0.387) on the same targets.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=tbgc772k&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
