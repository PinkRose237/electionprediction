# 2026 Midterm Forecast

A daily-updating forecast of the 2026 U.S. House, Senate and governor elections (Election Day: Nov. 3, 2026).
The model pulls polls, ratings, prediction markets, fundraising, economic data and news every day,
simulates every race 40,000 times with correlated errors, and rebuilds a static website.

**Live site:** https://pinkrose237.github.io/electionprediction/ (rebuilt every day at 12:00 PM Eastern by
[`.github/workflows/daily.yml`](.github/workflows/daily.yml), which also commits the day's forecast and history to this repo).
Start a run by hand from the repo's **Actions** tab ("Daily forecast" → "Run workflow").

## Run it locally

```bash
./run_daily.sh
```

That creates the virtualenv if needed, runs `python -m pipeline.run`, and writes:

| Output | What it is |
|---|---|
| `site/index.html` | The website, with the day's forecast embedded. Works from `file://`, any static host, or as a Claude artifact. |
| `site/data/forecast.json` | The same dataset as JSON. |
| `data/processed/forecast_YYYY-MM-DD.json` | Full model output for the day (every component, every poll). |
| `data/history/topline.csv`, `race_probs.csv` | Running history used for trend lines and the "since the last update" panel. |
| `data/raw/YYYY-MM-DD/` | Raw copies of everything fetched that day. |

Options: `--sims 20000` (faster), `--no-news` (skip headline collection), `--offline` (skip 270toWin).
To rebuild the page from the latest model output without refetching: `.venv/bin/python -m pipeline.export`.
To preview locally: `python3 -m http.server 8765 --directory site`.

The GitHub Action is the source of truth for the daily history, so run `git pull` before working locally and
don't commit history from local runs. A full run takes roughly 10–15 minutes, mostly because Wikipedia rate-limits large pages (the code honors its `Retry-After`).

## Sources (all fetched automatically each run)

| Source | Used for |
|---|---|
| Wikipedia (REST API) | Nominees, Cook PVI for all 435 districts, ratings from 10 forecasters (Cook, Sabato, Inside Elections, Split Ticket, DDHQ, The Economist, Silver Bulletin, FiftyPlusOne, RCP, Fox), race polls, published polling averages, approval aggregates |
| Silver Bulletin public poll sheet | Every generic-ballot poll |
| VoteHub API | Generic ballot, approval, Senate, governor and House polls |
| 270toWin | Senate and governor polls |
| Polymarket, Kalshi, PredictIt | Chamber control, race winners, national popular-vote ladders |
| FEC bulk file `weball26.zip` | Receipts and cash on hand for every House and Senate candidate |
| FRED | Unemployment, CPI, consumer sentiment, real disposable income, gas prices, GDP |
| Google News RSS | Headlines for every competitive race |
| FiveThirtyEight archive | ~20,000 historical polls with results, and pollster grades, used for calibration |
| Gallup (via UCSB) + Wikipedia | Presidential approval and House vote for every midterm since 1946 |

Hand-researched files live in `data/research/` (candidate experience and scandals, redistricting status, competitive House
districts, key events) and `data/manual/` (anything you add by hand, such as a poll the feeds missed, using the same CSV columns).

## Model

1. **National environment.** Generic-ballot polls (own adjusted average plus published averages, minus a historical
   bias correction), market-implied national vote, an approval-based fundamentals model fit on 1946–2022 midterms, and
   special-election swings, blended by inverse variance.
2. **Races.** Four components per race (fundamentals, polls, forecaster consensus, markets), combined by precision.
   Poll adjustments: partisan sponsors (−4.5 pts toward the sponsor; measured on the 538 archive), house effects,
   likely-voter shift, national trend; weights for recency, sample size, 538 grade and pollster crowding.
3. **Simulation.** National, office, regional and partisan-lean errors drawn jointly (fat-tailed Student-t), plus
   race-specific errors calibrated to the historical error of poll averages in competitive races.

Code map: `pipeline/fetch/*` (sources), `pipeline/model/polls.py` (poll averaging), `pipeline/model/national.py`
(national blend, fundamentals, backtest), `pipeline/model/races.py` (race components), `pipeline/model/simulate.py`
(Monte Carlo), `pipeline/run.py` (orchestration), `pipeline/export.py` + `site/template.html` (website).

## Margin of error

The site reports honest intervals rather than forcing them under a target. On Oct. 12 the national House-vote
estimate (D+7.5) has a standard error of 2.8 points: an 80% range of ±3.2 and a 95% range of ±5.6. A quarter of that
variance is time left before Election Day, so it narrows daily. In out-of-sample backtests on the 1998–2022 midterms,
the national method missed by 2.1 points on average. Individual races are less certain than the national vote: a
well-polled Senate race has an 80% range of about ±6 points, and an unpolled competitive House district about ±7.
