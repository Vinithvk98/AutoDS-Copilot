# AutoDS Copilot, Engineering Manual

A running record of how the copilot and trust features work, why they exist, and
where they live in the code. Written so a future you (or a reader of the polished
manual) can understand each part without reading the whole codebase. Each chapter
follows the same shape: the problem, what we built, why it is useful, how it
works, where to see it in the app, and how it is tested.

> Scope. This manual covers the upgrades layered on top of the original engine
> (the leakage-safe scikit-learn pipeline, the LangGraph agent, the MCP server,
> and the RAG knowledge base). The original engine is assumed and only referenced
> where the new work touches it.

## Contents

1. The LLM layer (provider-agnostic)
2. Grounded generation (no hallucination)
3. LLM wired across the copilot
4. Streaming answers
5. Instant pages with background AI enhancement
6. The conversational data copilot
7. Modelling rigor: imbalance, thresholds, calibration, fairness
8. The Class Balance proof step
9. Explainability and trust
10. Agentic actions (auto-pilot)
11. How it is configured and run
12. How it is tested
13. File map

---

## 1. The LLM layer (provider-agnostic)

**The problem.** The copilot needed to write prose (answers, insights,
summaries), but we did not want to hard-wire one vendor, leak API keys, or break
when no model is available.

**What we built.** A single module, `autods/llm.py`, that every piece of
generated text routes through. It supports three providers chosen by an
environment variable or auto-detected from whichever key is present:

- `openai` (needs `OPENAI_API_KEY`)
- `anthropic` (needs `ANTHROPIC_API_KEY`)
- `ollama` (a local model, no key, fully offline)

**Why it is useful.** One place to swap models, no vendor lock-in, and the tool
still runs with zero keys by falling back to rule-based text. Keys live only in a
gitignored `.env`, never in the code.

**How it works.**

- `provider()` resolves the active provider. If `AUTODS_LLM_PROVIDER` is set it is
  used, otherwise it auto-detects (OpenAI key, then Anthropic key, then a local
  Ollama server).
- `available()` is `True` when any provider is usable.
- `complete(system, user)` runs one non-streaming completion and returns text, or
  `None` on any failure, so callers can fall back. It never raises.
- `stream(system, user)` yields text chunks for the same completion (see chapter 4).
- Provider SDKs are imported lazily, so `openai` and `anthropic` stay optional.
- For a local model, `OLLAMA_KEEP_ALIVE` (default 15 minutes) keeps the model
  loaded between calls so each step does not pay the reload cost.

Settings live in `autods/config.py` (`LLM_PROVIDER`, `LLM_MODEL`,
`LLM_TEMPERATURE`, `LLM_MAX_TOKENS`), and a tiny dependency-free `_load_dotenv()`
reads `.env` at import so the file actually takes effect.

**How to see it.** `python3 -c "import autods.llm as l; print(l.describe())"`
prints the active provider and model, or `rule-based (no LLM configured)`.

---

## 2. Grounded generation (no hallucination)

**The problem.** An LLM left to itself invents numbers. For a data-science tool
that is fatal, a made-up metric is worse than no metric.

**What we built.** `autods/rag/generate.py`, the one place that turns retrieved
knowledge plus verified run facts into prose. Every generator shares a strict
system contract: use only the facts and notes provided, never invent numbers,
column names, or sources, and say when the context is not enough.

**Why it is useful.** Answers, insights, and summaries stay tied to real
retrieved material and real computed numbers, so they can be trusted and checked.

**How it works.** Functions include `answer()` and `answer_stream()` (the Ask
box), `insights()` (the pre and post modelling bullets), `rationale()` (a grounded
"why" for a recommendation), `executive_summary()`, and `compare_models()`. Each
builds a prompt from a facts block plus a passages block and calls
`llm.complete` or `llm.stream`. If no model is available they return `None` and
the caller uses rule-based text.

**How to see it.** Ask a question in the app, or read the insights on any step.
The wording is model-written, the numbers come from the run.

---

## 3. LLM wired across the copilot

**The problem.** We wanted the AI to help at every point the tool explains
itself, not only in a chat box.

**What we built.** The grounded generator is called from:

- `autods/rag/store.py` `answer()` and `rationale()`, the Ask answer and the model
  or cleaning recommendation "why".
- `autods/pipeline/insights.py` `pre_insights()` and `post_insights()`, the
  written analysis before and after modelling.

**Why it is useful.** The whole product reads like an analyst wrote it, and every
sentence is grounded. The raw retrieved knowledge base snippets still appear as
citation lines (prefixed with a book glyph) so sources stay visible.

**How it works.** Each function computes rule-based facts first (always correct),
then optionally hands those facts plus retrieved guidance to the LLM to rewrite.
The rule-based text is the fallback, so nothing breaks offline.

---

## 4. Streaming answers

**The problem.** A local model takes a few seconds to write an answer. Waiting for
the whole thing before showing anything feels slow.

**What we built.** Token-by-token streaming for the Ask box, so the answer paints
as it is generated.

**Why it is useful.** The answer feels instant and you can start reading
immediately.

**How it works.**

- `llm.stream()` yields chunks per provider (OpenAI `stream=True`, Anthropic
  `messages.stream`, Ollama line-delimited JSON). Empty on no provider or error.
- `generate.answer_stream()` returns that generator over the grounded prompt.
- `store.retrieve_for_answer()` splits retrieval out so the web layer can retrieve
  once, then stream.
- The Flask route `/api/ask_stream/<sid>` sends Server-Sent Events. Order matters:
  the `compute` card leads, then the answer streams as `token` events, then the
  `sources` come last (so the sources block does not sit under the answer growing
  as it fills, which caused a scroll glitch), then `done`.
- The browser (in `console.html`, `doAsk`) reads the stream and appends each token
  to the current chat bubble. Auto-scroll only follows when you are already at the
  bottom, so scrolling up to read mid-stream is not interrupted.

**How to see it.** Open Ask, type a question, watch it type back.

---

## 5. Instant pages with background AI enhancement

**The problem.** Once the LLM wrote the insights and rationale, every pipeline
page blocked on a model call (a few seconds each), so loading the sample and
moving between steps felt slow.

**What we built.** Pages render immediately with the fast rule-based text, then
the browser fetches the grounded LLM version in the background and swaps it in.

**Why it is useful.** Pages are snappy again, and the AI text still arrives, just
without making you wait for it. Re-visiting a step is instant because results are
cached.

**How it works.**

- Insight and rationale functions take a `use_llm` flag. The web layer passes
  `use_llm=False` for the instant render, so no page blocks on a model.
- Enhanceable regions carry a `data-enhance` attribute (for example
  `data-enhance="pre"`, `"post"`, `"why-models"`, `"summary"`, `"compare"`).
- After each render, `enhance()` in `console.html` fetches
  `/api/enhance/<sid>/<key>`, which computes the LLM version, caches it on the
  session, and returns the HTML to swap in. Offline it no-ops.
- A subtle "refining with AI" label shows while a region is enhancing.

**How to see it.** With a model configured, watch the insights and the executive
summary briefly show a "refining" state, then upgrade to the richer wording.

---

## 6. The conversational data copilot

**The problem.** The Ask box could answer general questions from the knowledge
base, but not questions about *your* dataset ("what is the churn rate by plan?").

**What we built.** A data-aware, multi-turn chat that answers questions about the
loaded dataset with the real numbers, safely.

**Why it is useful.** You can interrogate your data in plain language and get
answers computed from it, with the computation shown, so you can trust and repeat
it. It is the difference between a chatbot and a copilot that reads your data.

**How it works. The model never runs code.** It routes a question to one
whitelisted operation.

- `autods/dataqa.py` is a set of vetted pandas tools: `overview`,
  `column_summary`, `value_counts`, `group_stat` (grouped aggregate, for example
  churn rate by plan), `rate`, `correlation`, `top_correlations`. Each validates
  its arguments against the real columns, allows only a fixed set of
  aggregations, and returns compact JSON-safe numbers. Anything invalid returns an
  error, never an exception.
- `autods/converse.py` does the routing. `route()` asks the model to return a
  small JSON object naming one tool and its arguments (or `none` for a general
  question). The reply is parsed defensively and validated, so a bad or missing
  answer falls back to a general answer.
- `analyze()` runs the chosen tool and formats the result as grounded facts, which
  are handed to `generate.answer_stream()` so the spoken answer is built over the
  real numbers.
- The web route computes these facts, sends a `compute` card showing which tool
  ran and the figures, then streams the answer. Each turn is stored in
  `session["chat"]` so follow-ups ("what about by city?") have context.

**Safety.** Only whitelisted tools, only real columns, a fixed aggregation list,
no code execution, and it no-ops when no model is configured.

**How to see it.** Open Ask after loading a dataset and try "what is the churn
rate by plan" or "which features correlate most with churn". You will see an
"Analyzed your data" card with the computation and the numbers, then the answer.

---

## 7. Modelling rigor: imbalance, thresholds, calibration, fairness

Lives in `autods/pipeline/rigor.py`, with training changes in `train.py` and
reporting in `evaluate.py`.

### 7.1 Class imbalance (resampling)

**The problem.** When one class is rare (churn is about 15 percent), a model can
score high by always predicting the common class and never learning the rare one.

**What we built.** A `balance` strategy for training: `none`, `class_weight`,
`oversample`, or `smote`. When the data is imbalanced the app uses SMOTE, and if
`imbalanced-learn` is not installed it falls back to a dependency-free random
oversampler, so imbalance handling always works.

**Why it is useful.** The model sees both classes fairly and actually learns to
spot the rare, valuable case.

**How it works, and why it is leakage-safe.** The critical rule is **resample the
training split only, never the test split**.

- `train_model` splits first (`train_test_split`), setting the test data aside.
- `_fit_supervised` then resamples only `X_train, y_train`. For SMOTE it uses
  imbalanced-learn's `Pipeline`, which applies the sampler only during `.fit()`
  and does nothing at predict time. For oversample it upsamples the minority
  training rows with pandas.
- The test set keeps its real class ratio, so the reported metrics are honest.
- The strategy actually used is recorded on the result as `balance`.

Doing it the other way (resampling before the split, or resampling the test set)
is the classic mistake: synthetic minority points can leak from test into train,
or the score is measured on a fake balanced distribution. We avoid both.

### 7.2 Decision-threshold tuning

**The problem.** The default 0.5 probability cutoff is often wrong for imbalanced
data, giving high accuracy but missing most of the rare class.

**What we built.** `rigor.tune_threshold()` sweeps thresholds and reports the one
that maximizes F1, alongside the metrics at 0.5.

**Why it is useful.** On the sample it lifts F1 from about 0.26 to 0.37 and recall
from 0.19 to 0.44, a real gain in catching churners, just by choosing the cutoff.

### 7.3 Probability calibration

**The problem.** A model can be accurate yet output probabilities that do not mean
what they say (a "0.9" that is right only 60 percent of the time).

**What we built.** `rigor.calibration()` reports the Brier score and a reliability
curve.

**Why it is useful.** If you act on probabilities (rank customers by churn risk),
you need them to be honest. Brier is a single number for that, shown as a KPI.

### 7.4 Fairness checks

**The problem.** A model can be reliable overall but unfair or unreliable for some
groups.

**What we built.** `rigor.fairness()` computes per-group selection rate and
accuracy for each low-cardinality categorical feature and flags the largest gap.

**Why it is useful.** It surfaces bias to inspect (for example a much higher
positive rate for one city). It is framed as a signal to check, not a verdict.

All four findings are computed on real held-out data and appear in the
post-modelling insights and the dashboard.

---

## 8. The Class Balance proof step

**The problem.** Resampling happened invisibly inside training. There was no way
to *show* that imbalance was handled correctly.

**What we built.** A dedicated step, between model selection and training, that
displays the training-split class distribution before and after resampling, with
a two-panel chart (imbalanced on the left, balanced on the right), a numbers
table, and a note that the test set is left at its real ratio.

**Why it is useful.** It is live proof. If anyone asks "show me it is balanced",
you click into the step and there is the before and after, with numbers.

**How it works.**

- `train.balance_preview(df, target, strategy)` splits exactly as training does,
  then returns the class counts of the training split before and after resampling,
  plus the untouched test split, and a `changed` flag.
- The route `/api/balance/<sid>` runs the preview, renders the chart with
  `_balance_chart` (two side-by-side panels), and shows `partials/balance.html`.
- The model-selection and leaderboard forms route classification tasks to this
  step first; the step's button then trains the chosen model. Balanced data or a
  non-classification task shows "already balanced" or skips.

**How to read the chart.** Left panel is the training data as it came in, one
class much shorter than the other. Right panel is after resampling, both classes
equal. Left to right reads as problem then fix. The y-axis is training rows, the
x-axis is the class value.

---

## 9. Explainability and trust

Lives in `autods/pipeline/explain.py`, surfaced on the Results and Leaderboard
pages.

### 9.1 Permutation importance

**The problem.** Tree models expose importances, but many models do not, and the
importances are over encoded feature names that are hard to read.

**What we built.** `explain.permutation_importance()` measures how much the
held-out score drops when each *original* column is shuffled.

**Why it is useful.** It answers "what is the model actually using?" in readable
column names (`tenure_years`, `city`), works for any model (SVM, KNN, neural
nets), and would expose a model leaning on something it should not.

### 9.2 Error analysis

**The problem.** A single accuracy number hides where the model fails.

**What we built.** `explain.error_analysis()` reports the overall error rate and,
for classification, the categorical groups it gets wrong most (for example Miami
24 percent wrong versus Austin 4 percent). For regression it lists the largest
residuals.

**Why it is useful.** It tells you where not to trust the model and where to get
more data, instead of assuming it is equally good everywhere.

### 9.3 Executive summary

**The problem.** Stakeholders do not want a metrics table, they want the story.

**What we built.** `generate.executive_summary()` writes a grounded 3 to 5
sentence summary of the run from the verified facts (task, metrics, imbalance
handling, threshold, top drivers, error rate).

**Why it is useful.** It is the paragraph you send your manager or put atop a
report. It renders instantly as a plain sentence and upgrades to the AI version in
the background.

### 9.4 Model comparison reasoning

**The problem.** Picking the top score is not the same as choosing the right
model.

**What we built.** `generate.compare_models()` explains, over the leaderboard, why
the top model leads and the trade-off against a simpler or runner-up model.

**Why it is useful.** Model choice becomes a reasoned decision (accuracy versus
speed or interpretability), not just the biggest number.

### Where all of chapter 9 appears

Train a model and scroll the **Results** page: metric tiles (including the Brier
calibration score), then the **Executive summary**, then **What the model relies
on** (permutation importance), then **Where it makes mistakes**, then the
**Post-modelling insights** that also list these findings. The **Model
comparison** card is on the **Leaderboard** page under the table.

---

## 10. Agentic actions (auto-pilot)

**The problem.** The pipeline is a guided flow of manual steps. We wanted the
copilot to be able to *drive* it, proposing the decisions itself, while the user
stays in control.

**What we built.** An auto-pilot. From the explore step, "Let the copilot run it"
asks the agent to propose a full plan (which target to predict, which columns to
drop, which model to train), each with a short grounded reason. The user reviews
the plan and approves, then it runs end to end to the results page.

**Why it is useful.** It turns the tool from "recommend and you click" into
"the agent proposes a complete run and you approve it". It is the difference
between an assistant and an agent, and it is a strong thing to demo.

**How it works, and why it is safe.**

- Model choice is by analysis, not a default. Before proposing, the endpoint runs
  a real leaderboard (`leaderboard.run_leaderboard` cross-validates every
  candidate on the recommended-clean data). `autods/planner.py` `propose()` then
  picks the best performer for this dataset and explains it with the actual
  scores. So on the churn sample it picks GaussianNB (best F1), not a generic
  gradient boosting default. The plan page shows the top few models with their
  cross-validated scores and marks the chosen one.
- The agent gathers the schema, the recommended cleaning notes, the candidates and
  their scores, and asks the LLM for one structured JSON reply naming a model, the
  columns to drop, and a reason for each choice.
- Every choice is then validated. The model must be one of the real candidates,
  and the drop columns must be real feature columns. A bogus model or a
  hallucinated column is rejected and the recommended default is used instead. So
  the agent can never act on something that does not exist.
- With no LLM, the recommended heuristics (task detection, the cleaning plan, the
  default model) are used with plain reasons, so auto-pilot still works offline.
- The route `GET /api/autopilot/<sid>` builds the plan and shows
  `partials/autopilot.html`, a review card with the three decisions and their
  reasons, plus "Approve and run" and "Adjust manually". Nothing runs yet.
- Approval is client-side orchestration: `autorun()` in `console.html` reads the
  approved plan and drives the existing, already tested endpoints in order,
  `POST /api/models` to apply the cleaning, then `POST /api/train` to fit the
  chosen model, then renders the results. It reuses the same code paths a manual
  run would, so there is no separate, untested execution path.

**Human-in-the-loop.** The agent only ever proposes. Nothing is cleaned or trained
until the user clicks Approve, and "Adjust manually" drops back into the normal
guided flow at any point.

**How to see it.** On the explore step click "Let the copilot run it", read the
plan and its reasons, then Approve and run to land on the results page.

---

## 11. How it is configured and run

1. Optional model. Copy `.env.example` to `.env` and set one provider (an OpenAI
   or Anthropic key, or run a local Ollama). With nothing set, the tool runs fully
   offline on rule-based text.
2. Confirm the model is seen: `python3 -c "import autods.llm as l; print(l.describe())"`.
3. Run the console: `python3 run.py`, then open the local URL.
4. Environment variables: `AUTODS_LLM_PROVIDER` (auto, openai, anthropic, ollama,
   none), `AUTODS_LLM_MODEL`, `AUTODS_LLM_TEMPERATURE`, `AUTODS_LLM_MAX_TOKENS`,
   `OLLAMA_HOST`, `OLLAMA_KEEP_ALIVE`. Keys never get committed (`.env` is
   gitignored).

Optional Python packages: `imbalanced-learn` (SMOTE, falls back to random
oversampling), plus `openai` or `anthropic` for those providers. None are
required.

---

## 12. How it is tested

Everything runs offline in the test suite, with the model calls mocked, so no key
and no network are needed. The suite covers the LLM layer and grounded generation,
streaming, the instant-versus-enhanced behaviour, the data copilot tools and
routing, the SSE endpoint, resampling and the leakage-safe split, threshold
tuning, calibration, fairness, the balance step (before imbalanced, after
balanced, test untouched), permutation importance, error analysis, the executive
summary, and model comparison.

Run it with `python3 -m pytest -q`. Test files of note: `tests/test_llm.py`,
`tests/test_dataqa.py`, `tests/test_rigor.py`, `tests/test_explain.py`,
`tests/test_agentic.py`, plus the original `tests/test_pipeline.py`,
`tests/test_rag.py`, `tests/test_agent.py`.

---

## 13. File map

New modules:

- `autods/llm.py` provider-agnostic LLM layer (complete + stream).
- `autods/rag/generate.py` grounded generators (answer, insights, rationale,
  executive summary, model comparison).
- `autods/dataqa.py` safe pandas analysis tools for the data copilot.
- `autods/converse.py` question routing for the data copilot.
- `autods/planner.py` the agentic planner (propose a full run for approval).
- `autods/pipeline/rigor.py` resampling, threshold tuning, calibration, fairness.
- `autods/pipeline/explain.py` permutation importance and error analysis.

Touched modules:

- `autods/config.py` LLM settings and `.env` loading.
- `autods/rag/store.py` grounded and data-aware answers, retrieval split,
  rationale.
- `autods/pipeline/insights.py` grounded insights with rigor and explainability
  facts, instant vs enhanced.
- `autods/pipeline/train.py` balance strategies, leakage-safe resampling,
  `balance_preview`.
- `autods/pipeline/evaluate.py` rigor and explainability diagnostics.
- `autods/web/app.py` streaming endpoint, enhancement endpoint, balance step and
  chart, run-facts helper, data-aware Ask.
- `autods/web/templates/*` the chat UI, the balance step, the results and
  leaderboard cards, and small enhancement partials.
- `autods/web/static/style.css` styles for chat, balance, and explainability.

New web endpoints:

- `POST /api/ask_stream/<sid>` streamed, data-aware answers.
- `GET  /api/enhance/<sid>/<key>` background AI upgrade of a region.
- `POST /api/balance/<sid>` the class balance proof step.
- `GET  /api/autopilot/<sid>` the agent's proposed plan for approval.
