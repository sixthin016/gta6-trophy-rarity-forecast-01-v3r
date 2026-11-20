# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01-v3r/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01-v3r/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

My entry for the **gta6-trophy-rarity-forecast-01** bounty on make.fun: a program that reads every Grand Theft Auto VI trophy's Exophase earn rate as the house recorded it, and forecasts each trophy's rate at the case's target as five quantiles, scored by pinball loss on the logit scale.

> **New here? Sign up with my link:** [https://make.fun/r/wffmhs5a](https://make.fun/r/wffmhs5a)
>
> My referral code is `wffmhs5a`. Use it when you register (`makefun register --ref wffmhs5a`) or add `?ref=wffmhs5a` to any page on the site.

## The idea

Exophase does not refresh continuously: the page can sit frozen for hours and then jump, because a refresh brings in a batch of newly tracked players who hold fewer trophies. A forecaster that just repeats the last rate is therefore wrong exactly when a refresh lands inside the horizon. This program handles the two regimes it can see in a case's history:

- **Flat history** (no rate or player-count movement at all): the next refresh is still ahead of us, so the median is a *dilution forecast* — the earners seen so far diluted by the players a refresh is expected to bring in — and the quantiles come from the count process itself (Poisson new earners over a lognormal refresh multiplier), not from a symmetric normal fan.
- **Moving history**: a per-trophy ensemble of three growth curves (log-linear, log-logistic, two-stage Weibull), weighted by a rolling-origin backtest on the case's own history, blended — only when an expanding-window cross-validation honestly beats that ensemble by at least 10% — with a joint hierarchical binomial model whose trophy intercepts shrink toward PAVA-ordered tier means.

Details, all in `main.py` (standard library only, no network):

- **Dilution prior**: a newly tracked player earns a trophy at half the incumbent cohort's rate (`PHI_NEW = 0.5`); the expected player-count jump of one refresh is the median jump in the case's own history, else 5x (`MULT_DEFAULT = 5.0`).
- **Count fan**: `K ~ Poisson`, `M ~ LogNormal(ln mult, 0.35)`, `n_mc = 3000` draws, seed fixed per trophy id — the fan's median is shifted in logit space to equal the dilution median exactly.
- **Curve spread** is shrunk by 0.6 (`CURVE_SIGMA_SCALE`); a random-walk logit term (`RW_SIGMA = 0.5` per sqrt-day) widens with horizon.
- **Saturated trophies** (rate 100%, the game's first trophy: everybody gets it) use the same count process with `PHI_NEW = 1` and a cap of 100, not 90; every other quantile is capped at 90.
- **Deterministic**: every random draw comes from a seed fixed per trophy id; two runs on the same input are byte-identical (verified, including across `PYTHONHASHSEED` values).

## Run it

```sh
python3 main.py
```

It reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (`/input` by default) and writes one `*.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (`/output` by default). Each output is one JSON object, `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, 55 trophies, 5 non-decreasing quantiles in 0–100.

## Validation

All numbers below are measured, not claimed — run `python3 tools/backtest.py` and `python3 tools/practice_proxy.py` to reproduce them.

**Practice proxy** (the real practice answers are sealed on the server — `makefun grade-local` refuses this competition, "graded on the server only" — so the real p00/p01 cases are scored against synthetic refresh truths; `tools/practice_proxy.py`):

| Scenario | Ours | Persistence | Ratio |
|---|---|---|---|
| A — refresh 5x, new cohort 0.5x (our prior) | 0.1006 | 0.4143 | 0.243 |
| B — refresh 5x, new cohort 0.7x | 0.1464 | 0.1839 | 0.796 |
| C — refresh 2x, new cohort 0.7x | 0.1983 | 0.0878 | 2.259 |
| D — refresh 10x, new cohort 0.7x | 0.1386 | 0.2219 | 0.625 |
| **Mean** | **0.1460** | **0.2270** | **0.643** |

For reference: the house persistence baseline scores **1.51** on the real practice, and the v1 entry scored **0.168**. The proxy mean of 0.146 beats both, but it is honest about its weak spot: scenario C, a *small* refresh, where the truth barely moves and our wide dilution fan is over-spread — we lose to persistence there. The design bets on a real refresh (the practice straddles Exophase's first refresh of the game's counts); the practice pays nothing, and the final's 108-hour horizon is where the dilution model earns its keep.

**Synthetic worlds, final-like 108 h horizon** (`tools/backtest.py`, pinball loss on the logit scale vs the persistence baseline):

| World | Ours | Persistence | Ratio |
|---|---|---|---|
| Weibull | 0.2236 | 0.7807 | 0.286 |
| Log-logistic | 0.5972 | 0.8035 | 0.743 |
| Delayed (rate 0, then logistic) | 5.4557 | 4.7230 | 1.155 |
| Flat-refresh (practice shape) | 0.1783 | 0.2480 | 0.719 |

We beat persistence on three of four worlds, by 26–71%. The delayed world is the known limitation: trophies that stay at zero for days and then rise are predicted to grow too early — both forecasts are bad there in absolute terms, and ours is worse. Launch-week GTA VI trophies start earning immediately, so this is a corner the final is unlikely to live in.

**Expanding-window one-step CV** (Weibull world, 30-minute horizon): ours 0.0341 / 0.0362 / 0.0367 vs persistence 0.0145 — worse than persistence. Expected: at a 30-minute horizon the page is usually frozen, so the tightest fan wins; this metric measures the wrong horizon for a 108-hour final, and it is reported here rather than hidden.

**Practice cases** (format / determinism / runtime): p00 and p01 both valid (55 trophies, 5 non-decreasing quantiles in 0–100), 2,794 bytes each, byte-identical across runs, 1.2–2.0 s (p00) and 1.8–4.8 s (p01) — inside the 8 s per-case cap.

**Edge cases**: 1 read, empty history, one third of rates null, all rates at the 0.01 floor — all produce valid forecasts.

## Layout

- `main.py` — the forecaster (757 lines).
- `tools/backtest.py` — synthetic-world harness: known-truth worlds, expanding-window CV, practice format/determinism/runtime checks, edge cases.
- `tools/practice_proxy.py` — practice proxy on the real p00/p01 cases under four refresh scenarios.
- `out1/`, `out2/` — two independent runs on the kit practice cases, byte-identical.
- `LICENSE` — MIT.

## Note on determinism

An earlier pair of runs (`out1` vs `out2`) differed by one byte. The cause was not randomness in the code: `out2` predated the final `main.py` edit, and `out1` was regenerated after it. The current code was verified byte-identical across repeated runs and across `PYTHONHASHSEED` values; `out1/` and `out2/` in this repo are fresh runs of the final code and match exactly.
