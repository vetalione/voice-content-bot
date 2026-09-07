# Voice Content Bot — personal semantic editor

Existing Telegram → Groq Whisper → editor project, incrementally refactored.
The production text path uses explicitly selected OpenRouter models, including an
owner-selected free mode. Preserve ideas and relationships before choosing
which ones deserve a Threads draft, a Reel, or archive-only storage.

## Choosing models in Telegram

Send `/settings` (or `/models`) in the bot's private chat as `OWNER_TELEGRAM_ID`.
Telegram must subscribe to `callback_query` for buttons. At startup the bot adds
this subscription to an existing webhook matching `PUBLIC_BASE_URL` +
`WEBHOOK_PATH`, using the deployed `WEBHOOK_SECRET` and preserving pending updates.
It does not change another destination or a custom certificate. For manual setup,
run `.venv/bin/python scripts/set_webhook.py set` with the matching production
URL and secret in `.env`. Pending updates are preserved unless you explicitly
pass `--drop-pending`.
Choose free automatic routing (`openrouter/free`), a specific free text model from
the live catalog, or one of the configured paid Kimi models. The free list is
paginated and checks zero pricing; catalog presence does not guarantee current
quota or a working endpoint. No generation call is made when opening the menu.

The selection applies to **new recordings in both private chat and the channel**.
Each accepted job snapshots its settings before entering the queue. Changing the
menu does not change running/queued jobs or regenerate completed recordings.
`/status` shows the selection for new recordings. Only the owner can change it;
other allowed private users cannot change billing/model settings.

Free mode keeps the semantic pipeline, transcript checkpoints and bounded writer
selection, but disables paid fallback, escalation and Claude calls. Every free
request has zero-price routing constraints, with reasoning `low` where supported.
Quota exhaustion fails with the existing bounded retries/owner report; it never
switches to paid inference. Free generation is not guaranteed to handle every
long/dense recording: model context, output and daily quotas still apply.

Paid buttons use `OPENROUTER_PRIMARY_MODEL` and `OPENROUTER_ESCALATION_MODEL` and
require `OPENROUTER_ALLOW_PAID=true`. They retain the server's explicit writing
and audit policy. Turning off paid permission on the server disables an existing
paid selection too. A free selection can still run after the paid monthly budget
has been reached. No new environment variables or dependencies are needed.

Choices are saved in the existing checkpoint store (`voice_checkpoints` on
Supabase); apply `migrations/001_checkpoints.sql` first. Supabase preserves them
across Render redeploys. SQLite persists only while its disk persists; do not use
it for persistent Render Free settings. `CHECKPOINT_BACKEND=none` cannot save
model choices and the menu reports this instead of claiming success. Before the
first selection, existing environment configuration remains active.

## Architecture and reuse

Telegram whitelist/webhook, private forwarding, job runner/heartbeats, Groq Whisper,
ffmpeg preparation, long-audio chunks, overlap merge, absolute timestamps, delivery
and channel publication guard are reused. Legacy Groq/OpenRouter text clients and
old regression tests remain available with `SEMANTIC_PIPELINE_ENABLED=false`.

The semantic pipeline is:

1. Transcribe audio chunks once. Save each chunk and the merged transcript.
2. Pass A: deep extraction in 12-minute windows, with 2-minute overlap. A topic can
   contain primary, supporting, implication, counterpoint, story and synthesis atoms.
3. Continue overflow pages (at most two pages/window by default). Ten is a soft
   page guideline, never a hard total recording limit. Preserve overflow hints.
4. Pass B: global relationship/duplicate patch using the full transcript and atoms.
   Only conservative duplicates of the same semantic role are removed; source
   ranges and links are merged. Supporting/implication atoms are not their parent.
5. Pass C: independent primary-model coverage audit returns omissions/corrections,
   not a rewritten atom list. It also searches for conclusions across distant stories.
6. Optional one quality audit, selected by explicit rules below. Never run both
   Kimi escalation and Claude audits automatically on one recording.
7. Route **all** finalized atoms to THREADS, REELS, BOTH or ARCHIVE_ONLY.
8. Generate one teaser from the finalized understanding. Select at most four atoms
   each for Threads/Reels; writers receive relevant supporting atoms, implications,
   relationships, source excerpts and the editable voice guide.

All useful atoms, even ARCHIVE_ONLY, remain in the result and the owner's semantic
JSON document. Coverage confidence/density are subjective operational model judgments,
not scientific recall scores. Mocked tests verify contracts, not real model quality.

## Provider verification — 2026-09-07

A credential-free request to the live [OpenRouter catalog](https://openrouter.ai/api/v1/models)
returned both IDs with `expiration_date=null` and structured-output support:

| Model | Context | Input / million | Output / million | Cached input / million |
|---|---:|---:|---:|---:|
| `moonshotai/kimi-k2.5` | 262,144 | $0.45 | $2.25 | $0.07 |
| `moonshotai/kimi-k3` | 1,048,576 | $3 | $15 | $0.30 |

The previously reported September 7 removal is **not confirmed by this snapshot**.
A catalog entry does not guarantee a working endpoint, account credits, or future
availability. Startup refreshes the catalog; `/health.text_provider_health` shows
existence, selected primary/fallback, pricing and expiry, explicitly as a catalog
check. Cache refresh occurs on requests after five minutes.

No Kimi ID is hardcoded in Python logic. The primary must be configured. Missing
primary logs a clear error. Only an explicitly configured primary fallback can be
used after disappearance or HTTP 404, at most once per request. 429 never selects
another model. A more expensive fallback, or one whose price cannot be compared
with a missing primary, requires explicit escalation permission. Escalation is not
itself a primary fallback. There is no random free fallback.

Model capabilities select strict JSON schema, JSON Object Mode, or prompted JSON
in that order. Pydantic always validates. Invalid configuration/auth errors do not
silently downgrade. Generation repair is bounded to one retry; API 429 honors
Retry-After with at most two retries by default. Successful structured requests
are checkpointed, so repairs/resumes do not redo completed stages.

Reasoning is stage-specific: medium for extraction/merge/audit, low for writers.
If the catalog advertises a narrower effort set, use supported low instead of
silently increasing to maximum (K3 currently lists max/high/low). Unsupported
reasoning controls are omitted for non-reasoning models. Output caps include
reasoning; defaults are larger than in the old free-tier experiment.

## Exact configuration / Render migration

[.env.example](.env.example) is the complete credential-free template. `.env` stays
ignored. **Do not overwrite an existing `.env` with the template.**

Keep Telegram credentials and numeric whitelist/channel IDs, webhook/public URL,
Groq key/model, transcription language, audio chunking/runtime settings.
Add or explicitly update these values in Render:

```env
TRANSCRIPTION_PROVIDER=groq
GROQ_WHISPER_MODEL=whisper-large-v3
TEXT_PROVIDER=openrouter
SEMANTIC_PIPELINE_ENABLED=true
OPENROUTER_API_KEY=<existing OpenRouter key>
OPENROUTER_PRIMARY_MODEL=moonshotai/kimi-k2.5
OPENROUTER_PRIMARY_FALLBACK_MODEL=
OPENROUTER_ESCALATION_MODEL=moonshotai/kimi-k3
OPENROUTER_ALLOW_PAID=true
OPENROUTER_ALLOW_ESCALATION=true
QUALITY_AUDITOR=auto
FORCE_QUALITY_AUDIT=false
WRITING_PROVIDER=primary
EXTRACTION_WINDOW_MINUTES=12
EXTRACTION_OVERLAP_MINUTES=2
TARGET_ATOM_LIMIT=10
OVERFLOW_MAX_PASSES=2
SEMANTIC_MAX_INPUT_TOKENS=90000
SEMANTIC_MAX_OUTPUT_TOKENS=8000
SEMANTIC_REASONING_EFFORT=medium
WRITING_REASONING_EFFORT=low
ESCALATE_IF_DURATION_MINUTES=30
COVERAGE_CONFIDENCE_THRESHOLD=0.7
MAX_THREADS_CANDIDATES=4
MAX_REELS_CANDIDATES=4
THREADS_BATCH_SIZE=2
REELS_BATCH_SIZE=2
TEXT_TEASER_MAX_TOKENS=1800
TEXT_THREADS_MAX_TOKENS=5000
TEXT_REELS_MAX_TOKENS=6000
OPENROUTER_TIMEOUT_SECONDS=180
OPENROUTER_MAX_RETRIES=2
OPENROUTER_MAX_REQUESTS_PER_RECORDING=60
MONTHLY_LLM_BUDGET_USD=15
SOFT_BUDGET_WARNING_USD=10
STOP_ON_BUDGET_EXCEEDED=false
CHECKPOINT_BACKEND=supabase
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<backend service-role key>
CLAUDE_ENABLED=false
```

Blank optional values can be omitted. Model IDs above are explicit example
configuration, not application defaults. Leave fallback blank until choosing one.
Use a separate OpenRouter API key for this bot to isolate monthly usage reporting.

Obsolete on the semantic path: `OPENROUTER_MODEL`, `OPENROUTER_REASONING_EFFORT`,
`MINER_WINDOW_MINUTES`, `MINER_WINDOW_OVERLAP_MINUTES`, `TEXT_EXTRACTION_MAX_TOKENS`,
`GROQ_LLM_MODEL`, `GROQ_LLM_MAX_TOKENS`, `GROQ_LLM_TEMPERATURE`,
`GROQ_EXTRACTION_MAX_TOKENS`, `GROQ_MINING_MAX_TOKENS`, `GROQ_TEASER_MAX_TOKENS`,
`GROQ_THREADS_MAX_TOKENS`, `GROQ_REELS_MAX_TOKENS`, `GROQ_TPM_LIMIT`,
`GROQ_USE_JSON_SCHEMA`. These remain available to the legacy text path. The
semantic OpenRouter request budget uses `SEMANTIC_MAX_INPUT_TOKENS`; the legacy
OpenRouter path still uses `TEXT_MAX_INPUT_TOKENS`.
`GROQ_TIMEOUT_SECONDS` and Groq retry settings still apply to Whisper.

## Exact escalation policy

`QUALITY_AUDITOR=none` uses only the primary for extraction, global merge and audit.
`kimi_k3` selects the configured escalation model explicitly; `claude` selects
Claude explicitly. `auto` chooses the configured OpenRouter escalation model if
any of these evidence flags holds, and escalation is allowed:

- Duration >= `ESCALATE_IF_DURATION_MINUTES` (default 30).
- Coverage confidence < 0.7, unresolved omissions, or overflow.
- At least six estimated topic shifts.
- Several distant stories followed by a conclusion, but no synthesis detected.
- Broad atoms plus dense source or >=1,500 words.
- <=2 atoms, >=1,500 words, and >=3 topic shifts or dense content.
- `FORCE_QUALITY_AUDIT=true`.

Few atoms alone is not evidence of failure. Overflow also records when the final
set exceeds the soft target; all those atoms are retained. Signals are combined
conservatively from extraction and primary audit so early uncertainty is visible.

One premium audit stage per recording, with at most two structured attempts and
three HTTP attempts per structured attempt (six HTTP attempts worst case). There
is no recursive premium cascade. Optional quality audit failure leaves the primary
result intact, marks the auditor failed and warns the owner. Auto does not silently
switch from unavailable Kimi to Claude. All OpenRouter attempts also share the
recording cap, persisted across retries of the same Telegram file.

`WRITING_PROVIDER=primary` is the default. Explicit `escalation` or `claude` opts
final teaser/Threads/Reels writing into that provider, subject to the same paid/
escalation/enabled guards. It is not part of automatic escalation.

## Checkpoint storage and resuming

SQLite uses the standard library; Supabase uses existing httpx (no new base dependency).
For local development use `CHECKPOINT_BACKEND=sqlite` and
`CHECKPOINT_SQLITE_PATH=/path/to/checkpoints.sqlite3`.

For Render Free:
1. Create a Supabase Free project under your account.
2. Run [migrations/001_checkpoints.sql](migrations/001_checkpoints.sql) in its SQL editor.
3. Set `CHECKPOINT_BACKEND=supabase`, URL and backend service-role key in Render.
4. Never place that key in a browser, prompt, repository or public log.

The table has RLS enabled, no anon/authenticated access, and stores job identity,
status, validated LLM stages, chunk/full transcripts, analysis/results, publication
markers and provider usage metadata. Credentials are not checkpointed. Source file
identity is scoped to chat and private/channel mode. Reforwarding the same private
file with a new message ID reuses completed work and still never publishes.
Changed model/prompt/schema/budget inputs invalidate the corresponding LLM cache.

After a crash/redeploy, reforward the recording or run the private reanalysis CLI.
The queue itself is not durable and does not auto-requeue lost pending jobs.
Interrupted in-flight API calls may have been charged before a response was saved;
no system can reconstruct their result from this checkpoint. A crash between a
Telegram publication and saving its marker can still duplicate that publication.
Stored successful publication markers prevent ordinary replays from posting again.
Use one Render process and `JOB_CONCURRENCY=1`; this is not distributed locking.

[Render Free storage is ephemeral](https://render.com/docs/free); SQLite on its
local disk is not cross-deploy persistence. Supabase Free currently includes a
[500MB database](https://supabase.com/pricing) and may [pause after inactivity](https://supabase.com/docs/guides/platform/free-project-pausing).
Monitor database size and export/delete old records when needed. No external
Supabase project is provisioned automatically by this code migration.

## Claude Agent SDK setup

The optional official Python package is `claude-agent-sdk`. The verified current
model ID is `claude-sonnet-5` ([model documentation](https://platform.claude.com/docs/en/models/overview)).
The SDK runs a bundled Claude Code subprocess, not a stateless HTTP client.

For an explicitly enabled personal subscription workflow:

```bash
.venv/bin/pip install -r requirements-claude.txt
claude setup-token
```

Generate the token interactively with the official Claude Code CLI on your own
machine; do not paste the terminal output into chat or logs. Install the native
Claude Code CLI separately if the shell command is unavailable. Then set:

```env
CLAUDE_ENABLED=true
CLAUDE_MODEL=claude-sonnet-5
CLAUDE_CODE_OAUTH_TOKEN=<your token>
CLAUDE_TIMEOUT_SECONDS=300
CLAUDE_MAX_BUDGET_USD=0.5
QUALITY_AUDITOR=claude
```

[Claude Code authentication](https://code.claude.com/docs/en/authentication) documents
setup-token for noninteractive subscription usage, with a one-year token and a
Pro/Max/Team/Enterprise account. The [Agent SDK quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart)
primarily documents API/cloud auth and restricts offering subscription login/limits
to third-party product users without prior approval. This personal adapter is not
an entitlement to unrestricted hosted/SaaS subscription usage. Check your current
account/deployment terms; OAuth availability/limits must be verified on your account.
It is off by default and has not been live-tested with your subscription.
The [current subscription SDK notice](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
states that SDK and `claude -p` usage still draw from subscription limits (the
announced separate monthly-credit change is paused). Supabase's plan does not
affect this authentication. Render Free is a resource/reliability constraint,
not a prohibition on OAuth login; a local authenticated test is still required.

No `ANTHROPIC_API_KEY`, OpenAI key, alternate credential or cloud provider is used
as a fallback. Child overrides clear competing credential env variables; project
settings/hooks/MCP/tools are disabled and its configuration directory is isolated.
It receives only the stage prompt, runs at most two SDK turns and has a timeout.
SDK-reported `total_cost_usd` is an estimate for OAuth usage, not an OpenRouter charge
or proof that subscription use has no limits.

For a Docker build with this optional adapter:
`docker build --build-arg INSTALL_CLAUDE_SDK=true -t voice-content-bot .`
The normal Docker image excludes the SDK. Configure the build argument before
trying Claude on Render; setting `CLAUDE_ENABLED` alone cannot install a package.
The [SDK hosting guide](https://code.claude.com/docs/en/agent-sdk/hosting) suggests
1GiB RAM per fresh agent. Render Free may not have enough memory; local Claude
comparison against the same Supabase transcript avoids changing the bot's hosting.

## Private A/B comparison — no Whisper or Telegram calls

Use the recording ID printed in the owner's private report. Each command reads
the saved transcript, reuses compatible checkpoints and writes a separate result:

```bash
# Primary Kimi only, including its own coverage audit:
.venv/bin/python scripts/reanalyse.py --recording ID --auditor none --output /tmp/primary.json
# Primary plus explicitly configured Kimi K3 coverage audit:
.venv/bin/python scripts/reanalyse.py --recording ID --auditor kimi_k3 --output /tmp/k3-audit.json
# Primary plus Claude audit:
.venv/bin/python scripts/reanalyse.py --recording ID --auditor claude --output /tmp/claude-audit.json
# Compare extraction entirely on the explicitly selected K3 model:
.venv/bin/python scripts/reanalyse.py --recording ID --primary-model moonshotai/kimi-k3 --auditor none --output /tmp/k3-primary.json
```

These commands compare semantic analysis/omissions/structure; they do not publish
or automatically regenerate drafts. The resulting atoms/links can be inspected
side by side. Use `WRITING_PROVIDER` on a normal recording for explicit writer
comparison. Production whitelist checks remain upstream of all LLM work; unauthorized
private users are silently ignored. CLI usage assumes the trusted owner operating it.

## Requests and realistic budget estimates

With enough eligible atoms for four drafts of each kind, batches of two, no
repairs/overflow and normal speech density:

| Recording | Extraction windows | Primary calls incl. all writers | Optional premium audit |
|---|---:|---:|---:|
| 5 minutes | 1 | up to 9 | +1 only if triggered/forced |
| 15 minutes | 2 | up to 10 | +1 only if triggered/forced |
| 60 minutes | 6 | up to 14 | +1 in default auto mode |

Formula: windows + merge + primary coverage + routing + teaser + 2 Threads + 2 Reels.
Archive-only content needs fewer writing calls. Overflow can add one extraction
call/window; JSON repair/HTTP retry increases actual attempts. Short/dense speech
is not predicted by duration alone. Full transcript is used in global stages when
within configured/model context. An oversized global input fails clearly with saved
extraction retained; it is never silently truncated into a supposedly complete audit.

An illustrative hourly token assumption is **80k primary input + 14k primary
output**, plus **32k escalation input + 3.5k escalation output** when K3 audits
all long recordings. Output includes reasoning. At the verified prices above:

| Voice hours/month | Primary only | Including that K3 audit volume | With 50% planning margin |
|---|---:|---:|---:|
| 30 | $2.03 | $6.48 | $9.72 |
| 45 | $3.04 | $9.72 | $14.58 |
| 60 | $4.05 | $12.96 | $19.44 |

These are workload assumptions, not measured invoices or a guarantee. Many short
recordings each requesting eight drafts, dense speech, overflow, long reasoning
and retries can exceed them. Whisper, hosting and any Claude subscription/extra
usage are separate. Record real usage before promising $15 at 60 hours/month.

Every OpenRouter response records configured/actual model, stage, provider, tokens,
reasoning and returned cost. Per-recording usage is persisted even if JSON validation
then fails. Private diagnostics show model call counts, auditor, atom count and
reported cost; unknown cost is not invented. `cost_reports`/event coverage reveal
partial metadata. The monthly warning uses authoritative `/api/v1/key` `usage_monthly`
for this API key, not a fabricated sum presented as billing truth. Warning defaults
to $10; $15 is the target. Processing stops only when `STOP_ON_BUDGET_EXCEEDED=true`
and a pre-job check finds the budget exceeded; this is not an atomic dollar cap.
Use an OpenRouter key spending limit if you need a provider-enforced hard cap.

## Tests and ONE short smoke test

```bash
.venv/bin/pytest
.venv/bin/ruff check app tests scripts
```

Automated tests mock all providers. They cover the two required synthetic semantic
fixtures, preservation, overflow, quality policy, provider availability/capabilities,
OAuth isolation, persistent resume, usage, whitelist and private/public separation.

After setting the primary model, key and paid opt-in, run exactly this once:

```bash
.venv/bin/python scripts/smoke_semantic.py
```

This reads the public catalog and sends **one extraction request** for a tiny
synthetic Russian AI-art transcript. No Whisper, Telegram, escalation or retries.
It caps output at 4,000 tokens; at the captured K2.5 output rate the cap corresponds
to about $0.009 output plus the small input. It prints extracted claims for human
recall review. Passing validates one sample, not the whole pipeline or all models.
No live generation tests were run as part of this migration.

Registering the existing webhook again is unnecessary if the URL is unchanged.
The existing manual tools remain `scripts/set_webhook.py set` / `info`.
`DRY_RUN_PUBLISH=true` disables public posting. Private forwards always remain
private regardless of that flag. `VOICE_STYLE.md` remains user-editable.
