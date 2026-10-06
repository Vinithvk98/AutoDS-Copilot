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

Phase 1 is being built. See `autods/decision.py`, the `/studio` route, and
`templates/studio.html`.
