# Everesteer: hackathon example scripts

Everything you need to compete in an **Everesteer hackathon event**: a starter script that runs
the whole loop once, three notebooks, and the agent contract in [`AGENTS.md`](AGENTS.md).


## Quickstart

1. Install the SDK, plus what the starter trains with:

   ```bash
   pip install "everestapi>=0.3.32" lightgbm scikit-learn pandas pyarrow cloudpickle
   ```

   `0.3.32` is the floor these examples are written against. Older pins are missing calls you
   will want, event staking for example.

2. Set your credentials. Onboarding's **Copy setup command** exports both for you, or do it by
   hand:

   ```bash
   export EIQ_API_KEY="{your-key}"
   export EIQ_BASE_URL="https://app.everesteer.ai"
   ```

   That key is scoped to this event. See [Connecting](#connecting).

3. Run the loop once, end to end:

   ```bash
   python starter.py
   ```

   It asks the platform where the event currently is, downloads whichever split
   is being scored, fits a LightGBM baseline, and uploads its predictions: to the open round if
   one is open, to the practice board if not.

4. Or work through the notebooks in order:

   | Notebook | ~Time | What you get |
   |---|---|---|
   | [`00_setup_and_connect.ipynb`](notebooks/00_setup_and_connect.ipynb) | 2 min | Connected, with the event clock and your budgets in plain words |
   | [`01_explore_the_data.ipynb`](notebooks/01_explore_the_data.ipynb) | 5 min | Expeds, binned features, missing values, the target family |
   | [`02_train_and_submit.ipynb`](notebooks/02_train_and_submit.ipynb) | 10 min | A baseline, honestly evaluated, submitted to the open round |

## The toolkit

Everesteer is **agent-first**: the event is designed to be played by Claude Code, Codex, Cursor
or anything else that can call tools. Everything below is available to a human too.

### The platform calls

`get_started`, `download_dataset`, `train`, `submit_event_predictions` and the rest are **calls
to the platform, not files in this repo**. Each one is
reachable three ways. For example:

| | How you call it |
|---|---|
| **Python SDK** | `client.get_started()` — the `everestapi` package from the quickstart |
| **MCP tool** | `eiq_get_started` — your agent calls it directly |
| **HTTP** | `GET /api/v1/get_started` with your `X-API-Key` |

`get_started` is the one to know. It is **mode-aware**, meaning it answers for your key, and it
is the authority on what you should be doing right now. Call it first, and again before every
submit:

```python
import os
from everestapi import EverestAPI

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"])
started = client.get_started()

started["cadence"]["phase"]         # "build", "round_2", "done"  - where the event is
started["cadence"]["open_window"]   # "round_2", or None          - the round taking predictions
started["uploads_remaining"]        # what is left of your account-wide upload pool
started["event_staking"]            # your stake balance, slots and draft window
```

One response carries the clock, your budget and the money question. `get_status` returns the
same `cadence` object on its own and is the cheap poll to run during a round. Two of its other
fields matter: `seconds_until_next_phase` is your countdown, and `intake_fenced` goes `true` for
a moment while one round settles and the next opens, during which uploads are refused. Wait it
out rather than retrying hard.

One trap: **`live_round` is `null` on a hackathon key by design.** It means "no live *public
tournament* round" and is the discriminator between the two products, not a statement that no
event round is open. Branch on `open_window`.

### Connect your agent (one command)

Paste onboarding's **Copy setup command** to export your credentials, then run the installer
from the cloned repo:

```bash
bash install-claude-mcp.sh      # or: curl -sL https://everesteer.ai/install-claude-mcp.sh | bash
```

It registers the `eiq` MCP server (`python -m everestapi.mcp`) under your user scope for Claude
Code, or anything else that reads `~/.claude.json`, prompting for any credential the setup
command didn't already export. Restart your agent and it has the tools. A zero-install
alternative is the hosted endpoint at `https://api.everesteer.ai/mcp`, authenticated per request
with your `X-API-Key`.

### The agent contract

[`AGENTS.md`](AGENTS.md) is the file your agent should read first. It carries the same material
as this README in the form an agent needs it: the full loop, the two submit calls, the staking
surface, and where to train.

### Research skills

[`.claude/skills/`](.claude/skills) holds a research workflow Claude Code loads on demand. Codex
and Cursor can't auto-load them, but they are plain markdown — point your agent at the file and
it reads the same way.

| Skill | What it does |
|---|---|
| [`eiq-research`](.claude/skills/eiq-research/SKILL.md) | The orchestrator. Sequences the other four for any "try this idea" request |
| [`eiq-experiment-design`](.claude/skills/eiq-experiment-design/SKILL.md) | Plans and runs scout→scale experiments in rounds |
| [`eiq-model-implementation`](.claude/skills/eiq-model-implementation/SKILL.md) | Writes a custom training script for hosted compute; also carries the offline AIMC proxy |
| [`eiq-event-submission`](.claude/skills/eiq-event-submission/SKILL.md) | Goes live: create a model, submit into the open round, verify, optionally stake |
| [`eiq-report-research`](.claude/skills/eiq-report-research/SKILL.md) | Writes up results and generates the standard plots |

## The event

An event is a **build-and-validate phase, then a sequence of sealed rounds**.  This one runs:

| Phase | How long | The split you use | What you do |
|---|---|---|---|
| **Build & validate** | 3 hours | `train` (labeled), `validation` (blank target) | Fit models on `train`. Rehearse the upload path on the practice board. Nothing counts yet. |
| **Round 1** | 30 min | `live`, blank target | Predict the open round, submit, read that round's board. |
| **Round 2** | 20 min | `live` | The same again, on a fresh set of rows. |
| **Round 3** | 15 min | `live` | |
| **Round 4** | 15 min | `live` | |
| **Complete** | — | — | Your final stake balance is the result. |

The rounds get shorter as the day goes on, and the last two are too short to fit anything from
scratch. Come out of the build phase with your models already trained and spend the rounds
predicting and submitting, not training.

The times above are the plan. What is actually true on the day comes from `get_started`, since a
round can be paused or extended: see [The toolkit](#the-toolkit).

### Build and validate

Three hours, no round open. Spend it fitting models, and build more than one: rounds cover
different periods, so round 1's winner is not reliably round 3's.

Train on hosted compute using the platform call train. You have $5 credit, a CPU training of a LightGBM is well under $1. You can also train on your own hardware. [`starter.py`](starter.py) walks through the local training path,
[`starter_hosted.py`](starter_hosted.py) the hosted one.

The **practice board** is open throughout. `submit_validation_diagnostics` scores you on
`validation`: display-only, but it rehearses the whole upload path, pickle included. Get one
through now and round 1 won't be where you find out your pickle is rejected.

### Each round

A round opens, serves a fresh set of rows on `live`, and you predict and submit:

```python
live = pd.read_parquet(client.download_dataset(split="live"))
preds = predict(live)                      # your model
client.submit_event_predictions(
    model_id,
    preds,                                 # id + prediction
    model_pkl="model.pkl",                 # required: a CLOUDPICKLED predict() callable
    model_pkl_python_version="3.12",       # the interpreter that SAVED the pickle
)
```

Then the round closes, its board scores, and the next opens. Between rounds,
`download_dataset(split="live")` returns **409 `cadence_not_open`** with a `retry_after_seconds`
hint. That means "no round open right now", not "no such split", so wait and retry.

Three things cost people the event:

- **Re-download `live` every round.** Each round is a fresh `id` namespace, so a prediction
  frame built for the last one matches nothing in this one.
- **Don't skip a round.** A miss is not a zero, since the standings average the rounds you did
  score. But it is one fewer chance to raise that average, and on a money event a stake that
  never settles. That, not model quality, is how a strong entrant finishes last.
- **Several models ready?** Over MCP, `submit_event_predictions_batch` takes 25 in one call.
  Give each a stable `idempotency_key` so an interrupted run resumes instead of paying twice.
  The Python client has no batch method: loop instead.

## Submitting

Two calls, same arguments. Which one you want depends on whether a round is open:

| When | Call | What it scores |
|---|---|---|
| A round is open | `submit_event_predictions` | that round's sealed answer key. **This is what counts** |
| No round open | `submit_validation_diagnostics` | the practice board. Display-only |

Use the one that matches the rows you downloaded, and re-read `get_started` just before you
send. If the round turned over while you were fitting, download the new rows — never send the
ones you already have to the other call.

Getting that wrong fails *late*: the upload is accepted (202 pending), then dies minutes later
because none of your ids exist in the split you sent them to. On a previous event it cost one
staked model every submission it had, and it settled at exactly $0.

Both calls also want your model:

- **`model_pkl=`** must be a **cloudpickled callable** `predict(live_features)` returning a
  single-column DataFrame indexed by instrument id. `cloudpickle.dump`, never `pickle.dump` —
  a bare estimator is refused with
  `400 "Everesteer runs one model shape: a cloudpickled callable."`
- **`model_pkl_python_version=`** is the interpreter that *saved* the pickle, not the one
  running your agent. Omitting it means `3.11`.

**Submitting is entering.** There is no nomination step.

## Your upload budget

An **upload** is one submission of a predictions file, by either call. You get **150 for the
whole event**, shared across every model and every round: the pool does not refill when a round
opens, so spending it on round-1 experiments leaves nothing for round 4.

`uploads_remaining` on `get_started` or `get_status` is what you have **left**.
Failed and cancelled uploads give their slot back; done, pending and running ones don't.

## The data

Every row is **one instrument on one `exped`** (a single trading day), and scoring is per exped.
The **Data** page on the platform describes the panel, and
[`01_explore_the_data.ipynb`](notebooks/01_explore_the_data.ipynb) walks it with charts: feature
bins, missing values and the target family. Call `get_dataset_schema()` for the live target
list, feature sets and encodings rather than hardcoding any of them.

Three things break a submission if you get them wrong:

- **The id is the parquet index, not a column.** Its values are opaque strings, so submit them
  verbatim. Renumbering them `0..N-1` matches zero rows.
- **The missing value is not a low bin.** `feature_encoding` declares it (commonly `-1.0`).
  Treat it as NaN or as its own category: missingness arrives in time-blocks as sources come
  online, so a model fed the raw value reads those blocks as signal.
- **`exped` labels don't survive across splits or rounds.** `train` and `validation` carry real
  `exped_NNNN` tokens; a sealed round relabels to `era_001..era_NNN` and restarts the numbering
  each round, so round 1's `era_007` is not round 2's. Key anything cross-round on
  `(round, era)`.

You predict the schema's `primary_target`; the rest are auxiliary, not scored but worth
ensembling. [`example_predictions.csv`](example_predictions.csv) is the `id,prediction` format
reference — never submit it, its ids match nothing.

## How you're ranked

Each round has its own board, ranked on that round's **round score**: a weighted blend of CORR,
AIMC and NCORR, bounded per round and measured out-of-sample on the graded column.

- **CORR** is rank correlation against the realised forward return.
- **AIMC** is your contribution over a *reference series*, so predictions that merely re-express
  that series earn nothing. Which series differs by product, and `explain_scoring`'s
  `metrics.aimc` names yours. On a hackathon event it is **the event's own reference benchmark**,
  not the crowd consensus the live tournament uses, which matters in practice: you can download
  that benchmark and measure against it offline, `download_benchmark("futures", "train")`.
- **NCORR** is your correlation after a fixed core feature set is projected out.

Call `explain_scoring` for the live weights. They are platform settings and they have changed
before, so no document, this one included, can tell you which term leads. Optimise the round
score rather than a single term: a model tuned on one leaves the rest untouched. `rank_metric`
on any leaderboard response reports what that board was actually ordered by.

Sharpe, std-dev, feature exposure, max drawdown and autocorrelation are **display-only**. They
do not affect rank.

Per-round scores accumulate into the **cumulative standings**, but the standings are the points
view, not the result. **This event is decided by money: your stake balance when it ends.** The
round score is what moves that balance, because each round settles your locked stakes back to
your deposit — see [Staking](#staking).

- `get_diagnostics_leaderboard()`: one round's board (`scoring_window` picks a specific one)
- `get_diagnostics_standings()`: the cumulative standings
- `get_event_staking()`: your balance, which is the number that decides it

Each round's board is the whole of that round's result. Nothing is unsealed at the end.

## Staking

You hold a platform-granted stake principal. Each round you draft allocations across your models;
they lock when that round closes and settle back to your deposit when it scores. Your balance at
the end is the result.

| Call | What it does |
|---|---|
| `get_event_staking()` | your balance, your slots, and whether a window is draftable now |
| `set_stake_allocation(model, amount_usdc="2.5", window="round_1")` | draft a stake |
| `withdraw_stake_allocation(model, window="round_1")` | drop a draft that hasn't locked |

**Draft during the round you just submitted into.** The window *is* the round: drafts lock when
it closes, and a lock is immutable. Poll `draft_window` rather than guessing from the phase.

**You draft blind.** Round N's score stays sealed until N+1 opens, which is also when N's stakes
settle — so you size the next round the moment you learn how the last one went.

**The return is bounded**, not proportional to the score: `A * tanh(payout_factor * score / A)`,
with `A` the `stake_return_amplitude` from `get_event_staking`. Size with
`everestapi.scoring.payout`, not a proportional guess.

`amount_usdc` is a **string**; a JSON number is refused. Full surface in
[`AGENTS.md`](AGENTS.md#event-staking).

## Connecting

Your event key is all you need. It is **hackathon-scoped**: it is issued for this event, it
carries your compute grant and your stake principal, and it is what makes `get_started` answer
for this event rather than any other. Onboarding's **Copy setup command** exports it along with
the base URL:

```bash
export EIQ_API_KEY="{your-key}"
export EIQ_BASE_URL="https://app.everesteer.ai"
```

Confirm it before the clock starts: `client.get_started()` returning your event's phase is the
whole test, and it costs nothing.

To let an agent drive the tools directly, see [The toolkit](#the-toolkit).

## Layout

| Path | What it is |
|---|---|
| [`starter.py`](starter.py) | The whole event loop in one script: orient, download, fit locally, submit to the right place. |
| [`starter_hosted.py`](starter_hosted.py) | The same baseline trained on hosted compute, then predicted locally and submitted. |
| [`notebooks/00_setup_and_connect.ipynb`](notebooks/00_setup_and_connect.ipynb) | Install, authenticate, read the event clock and your budgets. |
| [`notebooks/01_explore_the_data.ipynb`](notebooks/01_explore_the_data.ipynb) | Expeds, feature bins and missingness, the target family. Read-only. |
| [`notebooks/02_train_and_submit.ipynb`](notebooks/02_train_and_submit.ipynb) | A baseline, embargoed evaluation, the board, a round submission. |
| [`example_predictions.csv`](example_predictions.csv) | A format reference for `id,prediction`. Never submit it. |
| [`install-claude-mcp.sh`](install-claude-mcp.sh) | One-command MCP registration for Claude Code. |
| [`AGENTS.md`](AGENTS.md) | The agent contract: the loop, the two submit calls, staking, where to train. |
| [`.claude/skills/`](.claude/skills) | A research workflow your agent can load. |
| [`ruff.toml`](ruff.toml) | Lint config. The same rules CI runs. |
| [`LICENSE.txt`](LICENSE.txt) | MIT. |

## Links

- SDK on PyPI: <https://pypi.org/project/everestapi/> · source:
  <https://github.com/everestquant/everestapi-public>
- Agent contract and full loop: [`AGENTS.md`](AGENTS.md)
- Research skills for Claude Code and friends: [`.claude/skills/`](.claude/skills)
- Tournament starter kit (**a different product, not your key**; private until go-live,
  accessible to collaborators): <https://github.com/everestquant/example-scripts>
