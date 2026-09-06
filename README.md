# Voice Content Bot

Turns a voice post in your Telegram channel into three things:

1. an **editorial teaser** published publicly under the original audio,
2. **Threads post drafts** (expert / professional positioning),
3. **Instagram Reels scripts** (talking-head, hook → payoff),

and sends 2 and 3 to you privately. Built for 15–20 minute recordings, tested up
to ~60 minutes.

Python 3.12 · FastAPI · aiogram 3 · Groq (STT + LLM) · ffmpeg · no database ·
zero additional monthly cost on free tiers.

---

## How it works

```
Telegram webhook ──▶ FastAPI  ──▶ aiogram router ──▶ dedupe + enqueue ──▶ 200 OK (≈15ms)
                                                            │
                                                            ▼
                                                   in-process JobRunner
                                                            │
  download ──▶ ffmpeg prepare/chunk ──▶ Groq STT ──▶ merge transcript
                                                            │
                                                content miner (windowed)
                                                            │
                            ┌───────────────────────────────┼──────────────────┐
                            ▼                               ▼                  ▼
                    channel teaser              Threads editor         Reels editor
                            │                               │                  │
                    publish to channel  ◀── channel mode only ──▶  private owner report
```

### Two entry modes

**Mode A — production channel.** A new `voice`/`audio` post appears in
`ALLOWED_CHANNEL_ID` → full pipeline → teaser published into the channel as a
reply to the original post (falling back to a plain new message if Telegram
refuses the reply) → Threads/Reels sent privately to `OWNER_TELEGRAM_ID`.

**Mode B — private test / archive.** Send or forward any voice/audio to the bot
in a private chat. Same pipeline, same code path, **nothing is ever published to
the channel**. Use it to test, to mine your old published voice posts, or to
recycle archive recordings.

Private mode is locked to `ALLOWED_USER_IDS` (the owner is always included).
Anyone else is dropped in the aiogram filter — no reply, no download, no Groq
call, no quota spent. The publish guard is a single check on
`JobRequest.may_publish`, enforced inside `ContentPipeline.publish`, so the two
modes cannot drift apart.

### Content mining, not summarising

The transcript is **not** summarised once and then reused. It is walked in
20-minute windows (`MINER_WINDOW_MINUTES`, 1-minute overlap) and each window is
mined for independent **content atoms** — ideas, frameworks, predictions,
opinions, paradoxes, personal stories, failures, travel incidents, provocative
questions. Each atom carries timestamps, a key claim, a verbatim supporting
quote, three independent 0–10 scores, and a confidence value. Atoms duplicated
across window seams are merged, keeping the higher-scored version.

Both editors then select from those atoms and are explicitly allowed to return
nothing rather than fill a quota.

---

## Project layout

```
app/
  main.py                  FastAPI app: /health, webhook endpoint
  config.py                every env var, in one Settings model
  container.py             composition root — the only place wiring happens
  models/                  Pydantic contracts between stages
    media.py               MediaRef, JobRequest, RecordingMetadata
    transcript.py          TranscriptSegment, Transcript, ChunkTranscript
    atoms.py               ContentAtom, ContentAtomSet, AtomCategory
    content.py             ChannelTeaser, ThreadsCandidate, ReelsCandidate
  telegram/
    handlers.py            routers; validate + enqueue only
    filters.py             the security boundary (channel id, user allow-list)
    downloader.py          Bot API getFile + 20 MB limit detection
    delivery.py            all outbound traffic (owner reports, channel publish)
    formatting.py          HTML escaping + safe 4096-char message splitting
    report.py              renders the private owner report
  services/
    audio.py               ffmpeg discovery, downsampling, overlapped chunking
    transcription.py       Groq STT → chunk transcripts with segments
    transcript_merge.py    timestamp offsetting + overlap de-duplication
    groq_client.py         the only AI provider; 429/5xx handling
    prompts.py             loads and renders prompts/*.md
    pipeline.py            orchestration
    processor.py           reporting, failure notification, transcript export
    jobs.py                in-process background queue
    dedupe.py              duplicate-update protection
  agents/
    base.py                structured JSON + one validation repair pass
    content_miner.py       windowing, atom extraction, atom de-duplication
    channel_teaser.py      public teaser
    threads_editor.py      Threads drafts
    reels_editor.py        Reels scripts
  utils/                   timecode, text normalisation, temp files, retry
prompts/                   editable markdown prompts + VOICE_STYLE.md
tests/                     115 tests, zero network calls
scripts/set_webhook.py     register / inspect / delete the webhook
```

Adding a database later touches exactly two seams: swap `TTLDedupeStore` for a
persistent `DedupeStore` implementation, and persist `PipelineResult` in
`RecordingProcessor`. Nothing else needs to change.

---

## Setup

```bash
git clone https://github.com/vetalione/voice-content-bot.git
cd voice-content-bot

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

touch .env
# then fill in BOT_TOKEN, ALLOWED_CHANNEL_ID, OWNER_TELEGRAM_ID,
# ALLOWED_USER_IDS and GROQ_API_KEY
```

`ffmpeg` is optional locally — `imageio-ffmpeg` (already in requirements) ships a
static binary that the app finds automatically. A system ffmpeg (`brew install
ffmpeg`) is still preferable because it brings `ffprobe`, which gives exact
durations.

### Getting the values

| Variable | Where from |
|---|---|
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) → `/newbot` |
| `OWNER_TELEGRAM_ID` | send anything to [@userinfobot](https://t.me/userinfobot) |
| `ALLOWED_CHANNEL_ID` | forward a channel post to @userinfobot, or read it from the app log after posting (`Ignoring post from unrelated chat -100…`) |
| `GROQ_API_KEY` | <https://console.groq.com/keys> |

Then add the bot to your channel **as an administrator with "Post messages"
permission** — without it the teaser cannot be published.

---

## Run locally

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
curl localhost:8000/health
```

`/health` reports whether config is complete and whether ffmpeg was found:

```json
{"status":"ok","configured":true,"missing_env":[],"ffmpeg":true,
 "queue":{"queued":0,"running":0,"completed":0,"failed":0}}
```

Telegram needs a public HTTPS URL for the webhook, so for local end-to-end
testing tunnel it:

```bash
ngrok http 8000                     # or cloudflared tunnel --url http://localhost:8000
# put the https URL into PUBLIC_BASE_URL in .env, then:
python scripts/set_webhook.py set
```

Now forward an old voice message to the bot in a private chat — that exercises
the whole pipeline without touching your channel.

Useful during local work:

- `DRY_RUN_PUBLISH=true` — run the full channel pipeline but publish nothing.
- `ENABLE_FULL_TRANSCRIPT=true` — also receive the transcript as a `.txt`.
- `KEEP_TEMP_FILES=true` — keep downloaded audio and chunks for inspection.
- `LOG_LEVEL=DEBUG` — see chunk planning, seam trimming and prompt decisions.

### Bot commands (private, allow-listed users only)

- `/start`, `/help` — what the bot does
- `/status` — queue depth, models, current audio limits

---

## Tests

```bash
pytest                     # 115 tests, ~3s, no network access
pytest -v tests/test_private_access.py
ruff check . && ruff format --check .
```

The suite covers: allowed vs disallowed channel; allowed vs unauthorized private
user; unauthorized users triggering **zero** processing, replies and Groq calls;
forwarded private voice/audio not publishing to the channel; unsupported post
types (text, photo, document, video note, edited posts); the content atom schema
(including score bounds, unknown categories, confidence-scale normalisation);
chunk timestamp offsetting; overlap merge and seam de-duplication; long Telegram
output splitting with balanced HTML; Groq failures (429 with bounded retries,
permanent 401, 5xx recovery, malformed JSON, network error) and the resulting
owner notification; and duplicate-update protection.

Every external dependency (Telegram transport, Groq HTTP, ffmpeg, filesystem
download) is faked. `Bot.__call__` is monkeypatched in the fixtures, so no test
can reach `api.telegram.org` even by accident.

---

## Deploy to Render (Free)

The repo is Docker-based so `ffmpeg` is installed via `apt`.

**Option A — Blueprint (recommended).** Push to GitHub, then in Render:
**New → Blueprint** and point it at the repo. `render.yaml` creates a free web
service with `healthCheckPath: /health` and generates `WEBHOOK_SECRET` for you.

**Option B — manual web service.**

| Setting | Value |
|---|---|
| Environment | Docker |
| Dockerfile path | `./Dockerfile` |
| Health check path | `/health` |
| Instance type | Free |
| Build command | *(none — Docker)* |
| Start command | *(none — the Dockerfile `CMD` handles it)* |

If you would rather use Render's native Python runtime (no Docker), ffmpeg still
works via the bundled `imageio-ffmpeg` binary:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`

Keep `--workers 1`: the job queue lives inside the process, so a second worker
would not see jobs submitted to the first.

### Required Render environment variables

Secrets — set these in the dashboard, never in git:

```
BOT_TOKEN
GROQ_API_KEY
ALLOWED_CHANNEL_ID
OWNER_TELEGRAM_ID
ALLOWED_USER_IDS
WEBHOOK_SECRET          (generated by render.yaml; any random string works)
```

Everything else has a working default. Worth setting explicitly:

```
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_LLM_MODEL=openai/gpt-oss-120b
TRANSCRIPT_LANGUAGE=ru
MAX_STT_UPLOAD_MB=24
AUDIO_CHUNK_MINUTES=10
AUDIO_CHUNK_OVERLAP_SECONDS=5
MAX_AUDIO_DURATION_MINUTES=75
MINER_WINDOW_MINUTES=20
ENABLE_FULL_TRANSCRIPT=false
LOG_LEVEL=INFO
JOB_CONCURRENCY=1
WORK_DIR=/tmp/voice-content-bot
```

See `app/config.py` for the complete list of settings and defaults.

### Register the webhook

After the first successful deploy, with `PUBLIC_BASE_URL` set in your local
`.env`:

```bash
python scripts/set_webhook.py set        # register
python scripts/set_webhook.py info       # inspect (shows last_error_message)
python scripts/set_webhook.py delete     # unregister
```

Or with plain curl:

```bash
curl -X POST "https://api.telegram.org/bot$BOT_TOKEN/setWebhook" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://<your-service>.onrender.com/telegram/webhook",
       "secret_token":"<WEBHOOK_SECRET>",
       "allowed_updates":["message","channel_post","edited_channel_post"]}'
```

---

## Design notes and limitations

### Webhook response time

The webhook handler parses the update, checks the dedupe key and pushes a job
onto an `asyncio.Queue` — measured at ~15 ms locally. Telegram never waits for
transcription. A 20–60 minute recording is processed by a background worker task
in the same process.

### Background jobs on Render Free — what you are accepting

The queue is an `asyncio.Queue` inside the web process. No Celery, no Redis, no
paid queue. The consequences are real and worth knowing:

- **Spin-down.** A Render Free service sleeps after ~15 minutes without HTTP
  traffic. Sleeping mid-job kills the job. A voice post published while the
  service is asleep still arrives: the incoming webhook wakes the service, but
  the first request after a cold start can take up to a minute, and Telegram
  retries for a while, which is why the dedupe guard exists.
- **Queue is not durable.** A redeploy or a crash loses queued and in-flight
  jobs. Nothing is retried automatically. When a job fails you get a Telegram
  message explaining why, and its dedupe key is released so re-forwarding the
  same recording to the bot in private mode reprocesses it.
- **Duplicate protection is process-local.** `TTLDedupeStore` is an in-memory
  TTL set (6 h, 2048 entries). It survives Telegram retries during a process
  lifetime, which is the stated v1 requirement, but not a restart.
- **Concurrency is 1.** Deliberate: it keeps you inside the Groq free-tier rate
  limits. A second recording posted while the first is processing waits in the
  queue; `JOB_QUEUE_SIZE` (16) bounds that, and an overflow notifies you rather
  than dropping the recording silently.
- **Free tier has ~512 MB RAM and an ephemeral disk.** Chunks are written to
  `WORK_DIR` and deleted in a `finally` block whatever happens.

Mitigation, if you want it: point an uptime pinger (e.g. a free cron-job service)
at `GET /health` every 10 minutes to keep the instance warm. That is the cheapest
way to make this reliable without leaving the free tier.

### Long audio and file sizes

Two independent limits are in play:

| Limit | Value | Handling |
|---|---|---|
| Bot API `getFile` download | 20 MB, hard | Detected before download; you get a clear message instead of a deep failure. Voice notes (opus) stay small — 60 min ≈ 8–14 MB. A forwarded 128 kbps mp3 hour is ~57 MB and cannot be downloaded by any bot. |
| Groq STT upload | `MAX_STT_UPLOAD_MB` (24) | Files that fit and are already in a Groq-supported container are uploaded untouched. Otherwise ffmpeg downsamples to 16 kHz mono mp3 @ 32 kbps (~14 MB/hour). Only if it is *still* too big does the file get split. |

Splitting uses overlapping chunks (`AUDIO_CHUNK_MINUTES`, default 10, with
`AUDIO_CHUNK_OVERLAP_SECONDS`, default 5) so no word is lost at a boundary. Chunk
length is additionally capped so each chunk fits the upload limit at the measured
bitrate, with a 15-second floor (below that, speech fragments too badly — you get
a warning telling you to lower the bitrate instead).

Chunks are transcribed independently, then merged: timestamps are shifted by each
chunk's offset so they refer to the **original** recording, segments that repeat
already-covered overlap text are dropped, the exact seam is trimmed token by
token, and timestamps are prevented from going backwards.

`MAX_AUDIO_DURATION_MINUTES` (75) is a safety valve — longer recordings are
rejected with an explanation rather than burning an hour of quota.

`mp3` is the default intermediate codec on purpose: `libmp3lame` is present in
every ffmpeg build, including the static `imageio-ffmpeg` one. Opus would be
smaller but is not guaranteed to be compiled in.

### Groq and cost safety

Groq is the only AI provider. There is **no** fallback to Anthropic, OpenAI or
anything else billable — a quota problem must show up as an error and a Telegram
message, never as a surprise invoice. Concretely:

- `429` is retried `GROQ_MAX_RETRIES` times (default 3) with exponential backoff,
  honouring `Retry-After`, then raised as `GroqQuotaError`.
- `5xx` is retried the same way.
- Any other `4xx` (bad key, unsupported file, malformed request) is permanent and
  raised immediately — retrying an invalid API key just wastes time.
- A job that ultimately fails sends you a Telegram message with the exception
  type and message, and publishes nothing.

Structured output: agents request `response_format=json_schema` derived from the
target Pydantic model, fall back to `json_object` if the model rejects the schema,
recover JSON wrapped in prose or a markdown fence, and on a validation failure
re-prompt **once** with the Pydantic errors attached. No important model response
is parsed with regex.

### Prompts

All prompts are markdown files in `prompts/`, loaded at runtime and cached per
process:

| File | Role |
|---|---|
| `content_miner.md` | how to find and score content atoms |
| `channel_teaser.md` | the public teaser: 2–5 sentences, no corporate language |
| `threads_editor.md` | selection rules and writing rules for Threads |
| `reels_editor.md` | Reels selection, beat structure, scripting rules |
| `VOICE_STYLE.md` | **your** voice — injected into all four of the above |

`VOICE_STYLE.md` is the highest-leverage file in the repo. It ships as a template
with sections for phrases you like, phrases you hate, tone, rhythm, profanity
policy, Threads examples, Reels examples, topics to emphasise and topics never to
publish. Two or three of your real posts pasted in there will change the output
more than any prompt rewrite. Prompts are cached per process, so restart or
redeploy after editing.

Placeholders use `{{name}}` so JSON examples and markdown braces inside prompt
files are left alone.

### Telegram message limits

Owner reports routinely exceed the 4096-character limit.
`split_html_message` cuts on the coarsest available boundary (blank line → line →
sentence → word), never inside an HTML tag, and closes and reopens any tags that
span a seam — so no part arrives with broken markup. The public teaser is sent as
plain text (no parse mode), so nothing your recording contains can break it.

---

## What you still have to do manually

1. Create the bot with @BotFather and copy `BOT_TOKEN`.
2. Add the bot to your channel as an **administrator with "Post messages"**.
3. Get `ALLOWED_CHANNEL_ID`, `OWNER_TELEGRAM_ID` and a `GROQ_API_KEY`.
4. Fill in `.env` locally, and the same variables in the Render dashboard.
5. Push the repo to GitHub and create the Render Free web service.
6. Run `python scripts/set_webhook.py set` once the service is live.
7. Fill in `prompts/VOICE_STYLE.md` — nothing else affects output quality as much.
8. Optionally: point an uptime pinger at `/health` to reduce cold starts.
