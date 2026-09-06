# Voice Content Bot

Telegram recordings become a channel teaser and selected Threads/Reels drafts.
Default providers: **Groq Whisper for speech-to-text, OpenRouter for all text**.
No `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is read or required.

## Processing and output limits

1. Download and prepare audio; transcribe each required audio chunk once with
   Groq Whisper. Text-stage retries never invoke Whisper again.
2. Extract compact atoms from 12-minute transcript windows, overlapping by 30
   seconds. A 60-minute recording normally uses six extraction windows. Input
   size checks may split unusually dense windows before requests are sent.
3. Deduplicate ideas globally, including repetitions at different timestamps.
4. Rank locally using extracted types, source availability and story duration.
   There is no LLM call to enrich every atom. The existing score fields remain
   neutral; they are not represented as model-generated editorial evaluations.
5. Create one teaser. Select at most four atoms for Threads and four for Reels
   **before** sending batches of two to the editors. Only those atoms are written.
   Results are aligned to source timestamps and ranked locally.

Extraction atoms contain only `title`, `start_seconds`, `end_seconds`, `idea`,
`type`, with up to eight atoms per window. Source excerpts are attached from the
already-produced transcript locally. The normal hour-long mocked regression
uses **11 LLM requests**: six extraction, one teaser, two Threads, two Reels.
This is not a guarantee for live routing: repairs and unusually large inputs can
add calls. A hard per-recording cap defaults to 30 OpenRouter HTTP attempts.

Channel posts can produce a public teaser; drafts/reports go to the owner.
Private/forwarded recordings are private tests and never publish in the channel.
`DRY_RUN_PUBLISH=true` disables public publication. `ENABLE_FULL_TRANSCRIPT=true`
also sends a transcript document to the owner.

## Local setup

Requires Python 3.12+. Docker includes ffmpeg; locally `imageio-ffmpeg` provides a
fallback binary. Existing dependencies already include httpx: no new dependency
is required for OpenRouter.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Fill in `.env`. Telegram channel/user IDs must be numeric; `ALLOWED_USER_IDS`
accepts comma-separated user IDs and the owner is included automatically.
`.env` and its variants are ignored; `.env.example` is a credential-free template.

## Render migration

Add these environment variables **before deploying the migration**:

```env
TRANSCRIPTION_PROVIDER=groq
TEXT_PROVIDER=openrouter
OPENROUTER_API_KEY=<your OpenRouter key>
OPENROUTER_MODEL=openrouter/free
OPENROUTER_ALLOW_PAID=false
```

Set/update:

```env
GROQ_WHISPER_MODEL=whisper-large-v3
MINER_WINDOW_MINUTES=12
MINER_WINDOW_OVERLAP_MINUTES=0.5
MAX_THREADS_CANDIDATES=4
MAX_REELS_CANDIDATES=4
THREADS_BATCH_SIZE=2
REELS_BATCH_SIZE=2
```

Keep `GROQ_API_KEY` for Whisper, Telegram credentials/IDs, webhook settings,
`PUBLIC_BASE_URL`, and audio/runtime settings. `GROQ_TIMEOUT_SECONDS` and Groq
retry settings still apply to transcription. They do not configure OpenRouter.

Remove or ignore these legacy Groq text-only settings on the OpenRouter path:
`GROQ_LLM_MODEL`, `GROQ_LLM_MAX_TOKENS`, `GROQ_LLM_TEMPERATURE`,
`GROQ_EXTRACTION_MAX_TOKENS`, `GROQ_MINING_MAX_TOKENS`, `GROQ_TEASER_MAX_TOKENS`,
`GROQ_THREADS_MAX_TOKENS`, `GROQ_REELS_MAX_TOKENS`, `GROQ_TPM_LIMIT`,
`GROQ_USE_JSON_SCHEMA`. Their implementation is retained for `TEXT_PROVIDER=groq`.
Do not add OpenAI or Anthropic keys.

Optional OpenRouter/text settings and defaults:

```env
OPENROUTER_REASONING_EFFORT=none
OPENROUTER_TIMEOUT_SECONDS=120
OPENROUTER_MAX_RETRIES=2
OPENROUTER_MAX_REQUESTS_PER_RECORDING=30
TEXT_MAX_INPUT_TOKENS=12000
TEXT_EXTRACTION_MAX_TOKENS=3000
TEXT_TEASER_MAX_TOKENS=800
TEXT_THREADS_MAX_TOKENS=2000
TEXT_REELS_MAX_TOKENS=3000
```

The existing Render Blueprint is updated. Existing dashboard overrides take
precedence over code defaults; explicitly update window and candidate settings.
Run a single process/worker: the job queue and deduplication store are in memory.
A Render restart/spindown loses queued/running jobs. Persistent job checkpoints
are not implemented by this provider migration.

## Free-only routing and structured responses

Requests go only to `https://openrouter.ai/api/v1/chat/completions`, authenticated
with `Authorization: Bearer OPENROUTER_API_KEY`. The configured model is sent
unchanged. With paid routing disabled, only `openrouter/free` or explicit
`:free` variants are accepted. All such requests additionally send zero maximum
prompt/completion/request/image prices. There is no paid model fallback list,
provider switch, credit top-up, or retry using a paid model ID.

A paid model requires BOTH an explicit model ID and `OPENROUTER_ALLOW_PAID=true`.
Keeping `openrouter/free` still applies zero-price constraints even with that flag.
Free availability/quota failures surface to the existing owner failure notifier
once applicable bounded retries have ended. Free routing does not guarantee
capacity or enough requests per day for every recording.

When a schema is supplied, request `json_schema` with strict output.
Direct model requests use `provider.require_parameters=true`. For `openrouter/free`,
use `require_parameters=false`: the router already performs feature selection,
and its generic parameter filter rejects the unified reasoning control with 404.
The schema and reasoning control are still sent, and Pydantic always validates
responses. This does not guarantee every upstream honors reasoning controls;
nonzero reported reasoning with reasoning disabled emits a warning. Only an explicit unsupported response-format error
allows a same-model, same-price-guard plain JSON request with the schema in the
instructions. A generic no-endpoints error does not downgrade or change models.
Requests explicitly disable reasoning by default using OpenRouter's unified
`reasoning: {"effort":"none","enabled":false}` control. Hiding reasoning with
`exclude:true` would still consume output tokens and is not used as a substitute.
`OPENROUTER_REASONING_EFFORT` can explicitly opt into minimal/low/medium/high.
Explicit incompatibility does not silently remove controls or switch to paid routing. An upstream may still reject or ignore a
control, so completion logs include finish reason and visible content length.
No Groq-specific reasoning settings or TPM scheduler run on this path.
See [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
Pydantic always validates results; invalid JSON/truncation and validation failures
have one local repair/retry. Invalid schema/authentication errors fail directly.
429 honors `Retry-After` (seconds or HTTP date), otherwise waits 60 seconds;
transient server retries are bounded and counted. Quota failures never recursively
split into many smaller requests.

References: [free router](https://openrouter.ai/docs/guides/routing/routers/free-router),
[structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs),
[provider pricing constraints](https://openrouter.ai/docs/guides/routing/provider-selection).

## Usage and diagnostics

Each OpenRouter attempt logs the recording's request count and stage. Responses
log the actual routed model, provider, input/output/reasoning tokens, and returned
cost when supplied. A final per-recording summary is emitted on success **and
failure**, with duration, request count, token totals, models and reported cost.
Missing metadata is not invented; `cost_reports` shows how many responses supplied
cost, so an incomplete sum must not be treated as an invoice or total estimate.

`/health` exposes safe process/worker diagnostics and configuration readiness.
Groq TPM diagnostics are populated only if the retained Groq text path is used.
Runtime logs are in Render's service **Logs**, separate from deploy build logs.

## Cheapest text-only smoke check

After filling only `OPENROUTER_API_KEY` in `.env`:

```bash
.venv/bin/python scripts/smoke_text.py
```

This defaults to `openrouter/free` with paid routing disabled and a 1024-token output cap,
and a maximum of **one HTTP request**, without retries. It validates a tiny
`{"ok":true}` response. It uses free quota, does not transcribe, and never sends
Telegram messages. A free model may still be unavailable or need more output
space; failure is reported without switching to paid routing. The smoke check is
manual and is never part of the test suite.

For a stronger check, use the real extraction prompt/schema on synthetic Russian
text (one HTTP request, no Whisper or Telegram):

```bash
.venv/bin/python scripts/smoke_text.py --extraction
```

Add `--model nvidia/nemotron-3-super-120b-a12b:free` to test that specific free
model instead of a random router selection. Paid model IDs are rejected. This
check uses the extraction output budget (default 3000) and validates nonempty
atoms. It is still not a full live recording/editor test.

## Tests and webhook

```bash
.venv/bin/pytest
.venv/bin/ruff check app tests scripts
```

All automated API tests use HTTP mocks; no real API calls. After deployment, an
existing webhook at the same URL continues working. For first-time registration:

```bash
.venv/bin/python scripts/set_webhook.py set
.venv/bin/python scripts/set_webhook.py info
```
