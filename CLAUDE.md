# CLAUDE.md — AutoDS Copilot

Context for any Claude session working in this repo. Read this first, then
`docs/MANUAL.md` for the deep technical record of every feature.

## What this project is

AutoDS Copilot turns a raw spreadsheet into a trustworthy, explained machine
learning model, with no code required. You point it at a CSV and it explores the
data, checks it for traps, cross validates dozens of models, trains the best one,
handles class imbalance honestly, and writes up the result in plain language. A
grounded AI copilot answers questions about the actual data, and an agent can plan
the whole run for you to approve. It runs locally and needs no API key by default.

AutoDS has two modes. Model Studio (the main app at /app) predicts, as above.
Decision Studio (at /studio, see docs/DECISION_STUDIO.md) is the no-code decision
intelligence mode: it auto builds a dashboard, finds what is notable, and
recommends a move with an estimated impact, with the same copilot over the data.
Phase 1 is built; the deeper phases (what-if, causal, optimization) are planned.

It is a portfolio project by Vinith Kumar that demonstrates a full, production
minded ML system: multi agent pipeline, RAG, MCP server, a grounded LLM copilot,
and genuine ML rigor (leakage safety, imbalance handling, calibration, fairness,
explainability).

## How to run and test

```
python3 -m pip install -r requirements.txt
python3 run.py            # web console at the printed local URL
python3 -m pytest -q      # full offline test suite (no key, no network)
```

Optional LLM: copy `.env.example` to `.env` and set one provider
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or a local Ollama). With none set, the tool
runs fully offline on rule based text. `.env` is gitignored; never commit keys.

## Architecture (where things live)

- `autods/agents/graph.py` the LangGraph pipeline of 9 specialist agents
  (loader, profiler, task detector, EDA, insight, cleaning, model recommender,
  trainer, evaluator) with human approval gates.
- `autods/mcp_server.py` exposes the pipeline as 9 MCP tools.
- `autods/pipeline/` the ML engine: loader, preclean, task_detect, eda, quality,
  preprocess, model_select, leaderboard, `train.py` (leakage safe pipeline +
  imbalance strategies + `balance_preview`), `evaluate.py` (metrics + rigor +
  explainability diagnostics), `rigor.py` (resampling, threshold tuning,
  calibration, fairness), `explain.py` (permutation importance, error analysis),
  `insights.py`, `report.py` (the written analysis report).
- `autods/rag/` retrieval: `store.py`, `knowledge.py`, `generate.py` (grounded
  LLM generators: answer, insights, rationale, executive summary, model comparison).
- `autods/llm.py` provider agnostic LLM layer (OpenAI, Anthropic, Ollama; complete
  and stream). `autods/config.py` settings and `.env` loading.
- `autods/converse.py` + `autods/dataqa.py` the conversational data copilot
  (routes a question to a safe pandas tool, answers from real numbers; also does
  multi-step why analysis that chains several tools, clarifying questions when a
  question is ambiguous, and suggested starter questions from the schema). Voice
  input is in the Ask UI via the browser speech API.
- `autods/planner.py` the agentic auto-pilot (cross validates, picks the best
  model, proposes a full plan to approve).
- `autods/decision.py` the Decision Studio engine (detects measure, dimension and
  time roles, then builds KPIs, a trend, breakdowns, key findings and a grounded
  recommendation with an estimated impact). Served by the /studio route and
  /api/studio, rendered by templates/studio.html with Chart.js.
- `autods/web/` Flask app. `app.py` routes, `templates/` (landing.html is a
  self contained redesigned marketing page; `console.html` is the SPA shell;
  `partials/` are the step views), `static/style.css` the shared design system.

## Conventions that matter

- HOUSE STYLE for all user facing copy (landing page, blog, report prose, UI
  text): no colons, no semicolons, no long dashes (em or en), no hyphens as
  connectors in prose, and no curly apostrophes or quotes. This is deliberate, it
  keeps the writing from reading as AI generated. Plain commas and periods only.
- Grounded generation only. The LLM may never invent numbers, columns, or sources.
  Everything it writes is built from retrieved knowledge plus the run's real
  computed facts, with a rule based fallback when no model is available.
- Leakage safety is sacred. Preprocessing and resampling happen on the training
  split only. The test set keeps its real distribution so scores stay honest.
- Tests run fully offline with the LLM mocked. Keep it that way. Do not use direct
  assignment like `llm.available = lambda: False` in tests, use monkeypatch, or it
  leaks into other tests.
- The sandbox cannot write to `.git`. Commits and pushes are done by the user from
  their Mac.

## Where to start now (read this)

Everything below is built, pushed, and tested (81 tests passing, run
`python3 -m pytest -q` to confirm). The immediate next task is **Decision Studio
Phase 2**. Open `docs/DECISION_STUDIO.md`, go to the "Phase 2 starting point"
section, and build from there. Keep everything grounded and in the house style,
and run the test suite as you go. Before coding, if there is any uncommitted work,
commit and push it.

## Current status (done)

Model Studio (the ML mode) core roadmap, all tested:
1. Conversational data copilot (ask questions of your real data, with multi-step
   why analysis, clarifying questions, suggested questions, and voice input).
2. Modeling rigor (SMOTE and oversampling, threshold tuning, calibration,
   fairness) plus a visible Class Balance proof step.
3. Explainability (permutation importance, error analysis, executive summary,
   model comparison).
4. Agentic auto-pilot (proposes and runs a full plan on approval, choosing the
   best model by cross validation).
Plus a grounded LLM layer with streaming, instant pages with background AI
enhancement, an expanded 10 section analysis report with inline diagnostic charts
(confusion matrix and calibration curve), a redesigned landing page, and a white
UI across all pages.

Decision Studio (the decision intelligence mode), Phase 1 done: the /studio route,
the auto dashboard (KPIs with period change, trend, breakdowns), key findings, a
grounded recommendation with an estimated impact, and the copilot over the same
data. Engine in `autods/decision.py`, page in `templates/studio.html`, tests in
`tests/test_decision.py`.

## Next steps

- **Decision Studio Phase 2 (do this next)**: what-if sliders, goal seeking, a
  polished executive share view. Then Phase 3 (causal, optimization). Full plan in
  docs/DECISION_STUDIO.md.
- Wrap the data copilot or the planner into the LangGraph graph or expose them as
  MCP tools (would grow the 9 and 9 counts).
- Add SHAP force or beeswarm plots as an optional explainability path.
- Give the About and Algorithms pages a hero header to match the landing page.

## Companion repo

The blog at `~/Desktop/GN2/blog` (github.com/Vinithvk98/vinith-notes) has a post
about this project at `autods.html`, kept in the same house style.
