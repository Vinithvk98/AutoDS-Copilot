# Decision Studio — the second mode of AutoDS

The blueprint for turning AutoDS from a single-purpose ML tool into a two-mode
product. Read `CLAUDE.md` and `docs/MANUAL.md` first for the existing system.

## The idea

AutoDS gains a second mode so it covers the whole arc from raw data to a decision
a business can act on, not just a trained model.

- **Model Studio** (the current app) predicts. Upload a CSV, explore, clean, cross
  validate, train, and explain a predictive model. Keep it exactly as it is.
- **Decision Studio** (new) decides. Point it at business data and it auto-builds a
  dashboard, lets you ask and speak questions, runs agentic why investigations,
  and ends with key findings and a recommended move with an estimated impact.

Inspired by Veezoo (natural language analytics and dashboards) and Graphite Note
(decision intelligence, the report to predict to decide chain). The goal is a
no-code tool a non-technical person can run and present to an executive.

## The decision chain (what Decision Studio produces)

Mirrors the Graphite framing, grounded and honest at each step.

1. **Report, what happened.** Auto dashboard. Detect measures, dimensions, and time
   columns, then show KPI tiles (with change versus the previous period when a time
   column exists), a trend line, and category breakdowns, with no setup.
2. **Explain, why.** Reuse the multi-step driver analysis from the copilot
   (`autods/converse.py` `multi_why`) to rank which dimensions move the metric most.
3. **Decide, what to change.** A recommended move plus an estimated impact computed
   from the data, clearly labeled as an estimate. For example, a segment far from
   the average, with the gap times its volume as the rough opportunity.
4. **Key findings.** An executive facing narrative of the notable points, grounded
   in the numbers, written in the house style.
5. **Present and share.** A clean, pitch ready view.

## What is reused (already built)

- Data loading and profiling (`pipeline/loader.py`, `preclean.py`, `quality.py`).
- The grounded LLM copilot, natural language and voice Q&A, multi-step why, and
  suggested questions (`converse.py`, `dataqa.py`, `rag/generate.py`, `llm.py`).
- The charting approach and the design system (`web/static/style.css`).
- The streaming Ask endpoint and the background enhancement pattern.

## What is new

- `autods/decision.py` the analytics engine for Decision Studio. Pure and grounded,
  no LLM needed. Detects measure, dimension, and time roles, then computes KPIs,
  a trend, breakdowns, key findings, and a grounded recommendation with an
  estimated impact.
- A JSON endpoint and a dashboard page (`/studio`) that renders KPI tiles, charts
  (Chart.js), key findings, the recommendation, and an Ask box over the same data.
- Mode entry points so a user chooses Model Studio or Decision Studio.

## Phased roadmap

**Phase 1, foundation (this session).** The mode switch, the auto dashboard (KPIs,
trend, breakdowns), key findings, and a first grounded recommendation. The copilot
Ask box works on the same data.

**Phase 2, decisions and depth.** Richer recommendations, what if sliders (move a
driver, see the estimated effect), goal seeking (what would it take to hit a
target), and a polished executive report and share view.

**Phase 3, causal and optimization.** Honest causal methods where the data supports
them (uplift, controlled comparisons) to separate real drivers from correlations,
and simple optimization to rank feasible actions by expected impact. This is the
research heavy part and comes last.

**Phase 4, connect and activate.** Beyond a single CSV, pull from a database or a
sheet, schedule refreshes, and deliver the recommendation out (an export or a
webhook). Optional, enterprise facing.

## Grounding and honesty

Every number on the dashboard and in the findings is computed from the data. The
recommendation's estimated impact is labeled as an estimate and shown with the
arithmetic behind it. We do not claim causal effect unless Phase 3 actually
measures it. Same house style as the rest of the product, no colons, semicolons,
long dashes, or curly quotes in user facing copy.

## Status

Phase 1 and Phase 2 are complete and tested.

- Engine: `autods/decision.py` (`build_overview(df)` returns measures, dimensions,
  time, kpis, trend, breakdowns, findings, recommendation).
- Web: the `/studio` page and `/api/studio/<sid>` endpoint in `autods/web/app.py`,
  rendered by `autods/web/templates/studio.html` with Chart.js. Entry points are on
  the landing page ("Explore decisions") and its nav.
- Tests: `tests/test_decision.py`.

To see it, run `python3 run.py` and open /studio, or click "Explore decisions" on
the landing page.

Phase 2, what was built.

- **What if.** `estimate_effect(df, measure, dimension, group, new_value)` and
  `estimate_effects(df, measure, dimension, changes)` in `decision.py`. Both rest on
  one exact identity. The overall mean is the row weighted average of the group
  means, so moving a group moves the overall by the change times the group's share
  of rows, and for a summed measure the overall total moves by the same amount as
  the group total. Groups do not overlap, so several moves add up. Rates are clamped
  to 0 to 100 percent. Endpoint `/api/studio/<sid>/whatif` (GET with dimension,
  group, value, or POST JSON with dimension and a changes map). On the dashboard a
  What if card has a slider per group for any of the top breakdowns, updates the bar
  chart live with a What if series, and shows the estimate with its arithmetic.
- **Goal seeking.** `goal_seek(df, measure, target)` (measure may be None for the
  primary). For each dimension with 2 to 12 groups it brings lagging groups toward
  the best per row level already reached in that dimension, biggest lever first,
  until the gap closes. It picks the option that moves the fewest groups, then the
  fewest points. No group is pushed past a level the data already shows, so when a
  target is out of reach it says so and reports the best reachable number. Endpoint
  `/api/studio/<sid>/goal`, surfaced as a Set a goal card.
- **Executive share view.** `/studio/<sid>/report`, template `studio_report.html`.
  The numbers, the recommendation, the goal plan when `?target=` is passed (the
  dashboard's share button adds the last goal automatically), key findings, the
  trend, and the top breakdown with a table, plus a Save as PDF button. Charts draw
  without animation so print captures them.
- **Richer recommendation.** The recommendation now knows direction. For measures
  where lower is better (churn, cost, delays, defects and similar, see
  `lower_is_better`) it targets the group furthest above the average and says bring
  it down, otherwise the group furthest below and says lift it. Before this it could
  suggest raising churn in the best plan. KPI tiles now lead with the primary
  measure. The overview also returns `primary`, `overall`, and per group `counts`.
- Tests in `tests/test_decision.py` check the estimates against the data recomputed
  after actually shifting the group, check that applying a goal plan reproduces the
  target, and check the copy for house style.

## Phase 2 starting point (done, kept for the record)

Build these three, in order, each grounded and in the house style, with tests, and
run `python3 -m pytest -q` as you go.

1. **What-if sliders.** On the Decision Studio dashboard, let the user move the
   primary measure for a chosen group (for example drag West's on-time rate up) and
   show the estimated effect on the overall number, live. The math already exists in
   `decision.py` `_recommend` (gap times the group's share). Factor that into a
   reusable `estimate_effect(df, measure, dimension, group, new_value)` function,
   add an endpoint `/api/studio/<sid>/whatif`, and wire a slider per group under the
   breakdown chart in `studio.html`. Label everything as an estimate.

2. **Goal seeking.** Let the user set a target for the primary measure (for example
   overall on-time rate of 60 percent) and compute what would have to change to get
   there, framed as the smallest feasible move across the top driver groups. Add
   `goal_seek(df, measure, target)` to `decision.py` and surface it as a small panel
   with the recommended changes and the resulting estimate.

3. **Executive share view.** A clean, print and present ready page (like the model
   report's "Open as a page") that lays out the KPIs, the trend, the top breakdown,
   the key findings, and the recommendation, with a Save as PDF button. Reuse the
   report page pattern. Route `/studio/<sid>/report`.

Keep the copilot Ask box working on the same session throughout. When Phase 2 is
done, update this file and `CLAUDE.md`, then move on to Phase 3 (causal methods and
optimization) from the roadmap above.

## Phase 3 starting point (do this next)

1. **Controlled comparisons.** Before calling a group a driver, check whether its gap
   survives holding a second dimension fixed (compare within each level, then
   reweight). Flag gaps that vanish as likely confounded. Add this to `decision.py`
   and show it next to the recommendation.
2. **Uplift where there is a treatment.** When a binary action column exists (for
   example promo in `sample_timeseries.csv`), estimate its effect with a simple
   difference in means with a confidence interval, then by segment.
3. **Optimization.** Rank feasible actions by expected impact per unit of effort
   using the goal seeking machinery, with a user set effort budget.
Label anything not measured as causal as an estimate, as now.
