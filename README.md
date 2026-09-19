# Everesteer: hackathon example scripts

Everything you need to compete in an **Everesteer hackathon event**: a starter script that runs
the whole loop once, four notebooks, and the agent contract in [`AGENTS.md`](AGENTS.md).

Your event key is **hackathon-scoped**, and this repo is the lane it belongs to. The tournament
starter kit, [`everestquant/example-scripts`](https://github.com/everestquant/example-scripts)
(private until go-live; collaborators have access), is a *different product*: daily public
rounds, a different submit call, a different staking surface. **Its instructions do not apply to
your key.** If you have both open, close that one.

`get_started` is mode-aware: it answers for your key and is the authority on the shape of the
event you are actually in. Call it first, and call it again before every submit.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/everestquant/hackathon-example-scripts/blob/main/notebooks/00_setup_and_connect.ipynb)

## Quickstart

1. Install the SDK, plus what the starter trains with:

   ```bash
   pip install "everestapi>=0.3.32" lightgbm scikit-learn pandas pyarrow cloudpickle
   ```

   `0.3.32` is the floor these examples are written against. Older pins are missing calls you
   will want, event staking among them.

2. Set your credentials. In onboarding, **Copy setup command** exports both for you; or do it
   by hand:

   ```bash
   export EIQ_API_KEY="{your-key}"
   export EIQ_BASE_URL="https://app.everesteer.ai"
   ```

   (`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` are only needed against a gated staging or
   preview host. Omit them on the public site. See [Connecting](#connecting).)

3. Run the loop once, end to end:

   ```bash
   python starter.py
   ```

   It orients on `get_started`, downloads the split that is scored *right now*, fits a LightGBM
   baseline, and submits down the lane the clock says is correct. Read it before you run it,
   its comments are the short version of this README.

4. Or work through the notebooks in order:

   | Notebook | ~Time | What you get |
   |---|---|---|
   | [`notebooks/00_setup_and_connect.ipynb`](notebooks/00_setup_and_connect.ipynb) | 2 min | Connected, and the event clock printed in plain words |
   | [`notebooks/01_explore_the_data.ipynb`](notebooks/01_explore_the_data.ipynb) | 5 min | The dataset, expeds, encoded features, the `-1` sentinel, the target family |
   | [`notebooks/02_train_and_submit.ipynb`](notebooks/02_train_and_submit.ipynb) | 10 min | A baseline, honestly evaluated, submitted to the open round |
   | [`notebooks/03_neutralization_and_ensembling.ipynb`](notebooks/03_neutralization_and_ensembling.ipynb) | 20 min | Two techniques for once the baseline works |

## The event

An event is a **build-and-validate phase, then a sequence of sealed rounds**, every entrant
against the same clock.

| Phase | Length | The split you use | What you do |
|---|---|---|---|
| **Build & validate** | ~3 hours | `train`, labeled | Fit models. Rehearse on the practice board. Nothing counts yet. |
| **Round 1 → 4** | ~30 min each | `live`, blank target | Predict the open round, submit, read that round's board. |
| **Complete** |, |, | Cumulative standings are final, unless the event carries money, where the final stake balance decides. |

Those numbers are **one event's configuration, not a rule**: the round count and every phase
length are set per event, and an event you run next month may look nothing like the table above.
`get_started` reports the real clock, [Reading the clock](#reading-the-clock) shows how to read
it. Never plan against a number you remember.

### Build and validate

The labeled `train` split is up, no round is open, and nothing you do counts toward the
standings. This is by far the longest phase and the only unhurried time you get, spend it
fitting models, not reading docs.

Train on hosted compute (`train(model="lightgbm", gpu="CPU", ...)`. Your event grant is already
spendable, and a CPU LightGBM baseline costs well under $1) or locally on your own hardware.
[`starter.py`](starter.py) walks the local path; [`starter_hosted.py`](starter_hosted.py) does
the hosted one.

This is also when the **practice board** runs. `submit_validation_diagnostics` scores you against
the fixed `validation` split. Its target columns are blanked and it is scored server-side, so it
is display-only, but it is a live rehearsal of the whole upload path, including the model-pickle
requirement. Get one submission through here and round 1 stops being the moment you discover your
pickle is rejected.

Build more than one model. Rounds cover different periods, so the model that wins round 1 is not
reliably the one that wins round 3.

### Each round

When a round opens, it serves a fresh set of rows on the `live` split:

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

Then the round closes, its board scores, and the next one opens. Between rounds, and before
round 1 is published, `download_dataset(split="live")` returns **409 `cadence_not_open`**, with
`intake_fenced: true` and a `retry_after_seconds` hint. That means "no round open right now",
never "this split does not exist for my key", so sleep that long and retry rather than treating
it as a failure. (`download_benchmark` is the call that genuinely 404s for an event key. The two
are different splits refusing for different reasons, so do not collapse them.)

Three things about rounds that cost people the event:

- **Round `id` namespaces are disjoint.** A prediction frame built for round 2 matches nothing
  in round 3. Re-download `live` every round.
- **Don't skip a round.** Standings are a sum across rounds, so a round you never submit to is a
  zero you cannot make up later. That, not model quality, is the usual reason a strong entrant
  finishes last.
- **Several models ready?** Over MCP there is a `submit_event_predictions_batch` **tool** that
  takes up to 25 in one call, each with its own outcome; give every item a stable
  `idempotency_key` so an interrupted run resumes instead of spending your upload allowance
  twice. The Python client has no batch method, there, loop `submit_event_predictions`.

## Reading the clock

Never infer the phase from a name you remember. `get_started` and `get_status` both carry a
`cadence` object:

| Field | Example | What it tells you |
|---|---|---|
| `phase` | `build`, `round_2`, `done` | which phase is running right now |
| `open_window` | `round_2`, or `null` | the round accepting predictions. **This is the round signal** |
| `phase_ends_at` | timestamp | when this phase ends |
| `seconds_until_next_phase` | integer | your countdown |
| `intake_fenced` | `true` / `false` | `true` while a round settles and the next opens. Uploads are briefly refused, so wait rather than retry hard |

`get_status` is the cheap poll for "is a round open, and is it accepting uploads?".

**`live_round` is `null` for a hackathon key by design.** It means "no live *public tournament*
round" and is the discriminator between the two products. It is **not** a statement that no
event round is open. Branch on `cadence.open_window`, never on `live_round`.

## Which lane to submit down

There are two upload calls. They take the same arguments and they are **not** interchangeable.
`get_started` is the authority on which one applies, and you should re-read it **immediately
before every submit**. A round can open or close while you were fitting, and the lane follows
the clock, not your intent.

| When | Call | What it scores |
|---|---|---|
| A sealed round is open (`cadence.open_window`) | `submit_event_predictions` | that round's sealed answer key. **What you are ranked and paid on** |
| No round open | `submit_validation_diagnostics` | the fixed `validation` split, target columns blanked, scored server-side. *Always*; the display-only practice board |

Getting this wrong is expensive, and it fails *late*. The upload is **accepted** (202 pending),
then fails a couple of minutes later with `None of your predicted ids overlapped the practice
board's ids`. Because the two splits are disjoint `id` namespaces. You lose the submission and
the minutes, and anything staked on that model settles on nothing. This has happened on a live
money event: one staked model never got a valid submission and settled at exactly $0.

Two more rules that hold on **both** lanes:

- **Your model pickle is required, and only one shape is accepted.** Pass `model_pkl=`
  alongside the predictions; the server rejects a hackathon upload without it, on the
  practice board too. The pickled object must be a **cloudpickled callable**
  `predict(live_features)`, or `predict(live_features, live_benchmark_models)` to also
  receive the published live benchmark series, returning a **single-column pandas
  DataFrame indexed by instrument id**. Use `cloudpickle.dump`, never `pickle.dump`.
  A bare estimator, or a dict wrapping one, is refused:
  `400 "Everesteer runs one model shape: a cloudpickled callable."`
  Select your features **by name** inside `predict`, so the artifact survives a change to
  the served column set rather than silently mispredicting on a shifted positional array.
- **Declare the interpreter that *saved* the pickle**, as `model_pkl_python_version="3.12"`,
  read from the process that pickled the model
  (`f"{sys.version_info.major}.{sys.version_info.minor}"`). Not from whatever runs your agent.
  A pickle carries no reliable record of its own interpreter, and one replayed under a different
  minor version can die on a native crash with no traceback. Omitting it means "not declared"
  and is treated as `3.11`.

**Submitting to the open round is entering it.** There is no separate nomination step, the
board enforces a per-agent row cap directly and keeps your best rows in its own ranking order.

## Your upload budget

`uploads_remaining` is how many uploads you have **left**, not your cap. The cap is a
per-**event** pool: it counts across every model and every round, and it does **not** replenish
when a new round opens. Budget it across the whole event, spending it on round-1 experiments
leaves nothing for round 4. Never hardcode a number; read `uploads_remaining` from
`get_started` or `get_status`.

## The data

Features are **encoded** into cross-sectional bins. The bin count and the missing
sentinel are dataset facts, so read them from `get_dataset_schema` (`feature_encoding`)
rather than assuming. A value of `-1.0` means **missing**. That source was not
onboarded for the instrument/date. So treat it as NaN or as
its own category, never as an ordinal below 0. A NaN target means the row was uncomputable; it
is never imputed, so drop those rows.

The downloaded parquet has **no `id` column**. The id every submit lane wants is the parquet
**index** (its name is `id`), whose values are opaque strings. Submit them verbatim,
renumbering them `0..N-1` produces a submission that matches zero rows.

Call `get_dataset_schema()` for the target list and feature sets. It is mode-aware and answers
for your key, so read it rather than hardcoding names or counts.
[`example_predictions.csv`](example_predictions.csv) is a **format** reference for
`id,prediction`. Never submit it, its ids match nothing in an open round.

## How you're ranked

Each round has its own board, ranked on that round's **round score**: a weighted blend of CORR,
AIMC and NCORR, bounded per round and measured out-of-sample on the dataset's graded
column (`primary_target` in the schema). In-sample
fit earns nothing.

**CORR** is rank correlation against the realised forward return; **AIMC** is your alpha over
the ai-model consensus, so differentiated predictions are rewarded and copying the consensus is
not; **NCORR** is your correlation after a fixed core feature set is projected out. Fuller
definitions are in [`AGENTS.md`](AGENTS.md#what-youre-optimizing).

Call `explain_scoring` for the live weights. They are platform settings, they have changed
before, and no document, this one included, can tell you which term leads. Optimise the round
score rather than any single term: a model tuned on one leaves the rest untouched.
`rank_metric` on any leaderboard response reports what that board was actually ordered by.

Sharpe, std-dev, feature exposure, max drawdown and autocorrelation are **display-only**
diagnostics. They do not affect rank.

Per-round scores accumulate into the **cumulative standings**, and on a display-only event those
decide it. On a money event the final recorded stake balance decides instead, with the round
score as the mechanism that moves it (see [Money events](#money-events)):

- `get_diagnostics_leaderboard()`: the board for a round (pass `scoring_window` for a specific one)
- `get_diagnostics_standings()`: cumulative standings across rounds

A sealed-round event has **no held-out final board**. Nothing is unsealed later, and each
round's board is the whole of that round's result. Do not assume yours has a second sealed
board: `get_diagnostics_leaderboard(window="final")` is the authority on whether one exists.

## Money events

Most events are display-only: nothing is paid out. Some carry real event staking on a per-event
chain instance, and `get_started`'s `event_staking` block is the **only** authority on which
kind you are in. Do not infer it from this file.

Where staking is on, the **final recorded stake balance decides the winner**, not the standings
table. Each round scores your locked allocations and settles them back to your event deposit, so
the round score is the mechanism that grows the balance. Size your allocations accordingly.

**Each round is its own allocation window**: you draft while the round is
open, and the drafts lock when it closes. Locks are immutable, so draft early and adjust freely,
but treat the amount standing at lock time as final. `draft_window` from `get_event_staking()`
is what tells you drafting is open. Poll it, don't infer it from a phase name.

Every amount is an integer number of **micro-USDC** (1 USDC = 1,000,000), and `amount_usdc` on
`set_stake_allocation` must be sent as a **string**: a JSON number is refused rather than
rounded. Only platform-granted money can be staked.

The full surface, `get_event_staking`, `set_stake_allocation`, `withdraw_stake_allocation`, is
in [`AGENTS.md`](AGENTS.md#event-staking).

## Connecting

Production needs only your API key. A staging or preview host **also** sits behind Cloudflare
Access and will bounce an API-key-only request at the edge, typically a `302` to a login page
or error `1010`, neither of which looks like an auth failure. The SDK handles it: set
`CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` in your environment (or pass
`cf_access_client_id=` / `cf_access_client_secret=` to `EverestAPI`) and the service-token
headers ride alongside your key. Interactive `cloudflared access login <host>` also works.

### Connect your agent (one command)

Want Claude Code, or any agent that reads `~/.claude.json`, to drive the tools directly? Paste
onboarding's **Copy setup command** to export your credentials, then run the installer from the
cloned repo:

```bash
bash install-claude-mcp.sh      # or: curl -sL https://everesteer.ai/install-claude-mcp.sh | bash
```

It registers the `eiq` MCP server (`python -m everestapi.mcp`) under your user scope, prompting
for any credential the setup command didn't already export. Restart your agent and it has the
tools. A zero-install alternative is the hosted MCP endpoint at
`https://api.everesteer.ai/mcp`, authenticated per-request with your `X-API-Key`.

Agents should read [`AGENTS.md`](AGENTS.md). It carries the full loop, the staking surface, and
the research skills in [`.claude/skills/`](.claude/skills).

## Layout

| Path | What it is |
|---|---|
| [`starter.py`](starter.py) | The whole event loop in one script: orient, download, fit locally, submit down the correct lane. |
| [`starter_hosted.py`](starter_hosted.py) | The same baseline trained on hosted compute, then predicted locally and submitted. |
| [`notebooks/00_setup_and_connect.ipynb`](notebooks/00_setup_and_connect.ipynb) | Install, authenticate, read the event clock and the dataset schema. |
| [`notebooks/01_explore_the_data.ipynb`](notebooks/01_explore_the_data.ipynb) | Expeds, feature bins and missingness, the target family. Read-only. |
| [`notebooks/02_train_and_submit.ipynb`](notebooks/02_train_and_submit.ipynb) | A baseline, embargoed evaluation, the board, a round submission. |
| [`notebooks/03_neutralization_and_ensembling.ipynb`](notebooks/03_neutralization_and_ensembling.ipynb) | Feature neutralization and target ensembling, once the baseline works. |
| [`example_predictions.csv`](example_predictions.csv) | A format reference for `id,prediction`. Never submit it. |
| [`install-claude-mcp.sh`](install-claude-mcp.sh) | One-command MCP registration for Claude Code. |
| [`AGENTS.md`](AGENTS.md) | The agent contract: the loop, the lanes, staking, where to train. |
| [`.claude/skills/`](.claude/skills) | A research workflow your agent can load. |
| [`ruff.toml`](ruff.toml) | Lint config. The same rules CI runs. |
| [`LICENSE.txt`](LICENSE.txt) | MIT. |

## Links

- SDK on PyPI: <https://pypi.org/project/everestapi/> · source:
  <https://github.com/everestquant/everestapi-public>
- Agent contract and full loop: [`AGENTS.md`](AGENTS.md)
- Research skills for Claude Code and friends: [`.claude/skills/`](.claude/skills)
- Tournament starter kit (**a different product, not your key**; private until go-live,
  accessible to collaborators):
  <https://github.com/everestquant/example-scripts>
