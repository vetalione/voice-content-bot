# Semantic editor migration plan — 2026-09-07

Reuse Telegram whitelist/webhook/job runner/private publication guard, Groq Whisper,
audio preparation/chunking, timestamp merge, Pydantic, existing delivery and editors.

Incremental implementation:
1. Configurable primary/fallback/auditor clients; discover models and capabilities.
2. Persistent validated request checkpoints plus transcription and final results.
3. Deep extraction, overflow continuation, global graph merge, omission-only audit.
4. Evidence-based bounded premium audit; preserve every atom and route writing separately.
5. Private diagnostics, provider-reported budget usage, saved-transcript A/B CLI.
6. Mocked regressions, full old suite, one manual short smoke instruction.

Public OpenRouter /api/v1/models queried without credentials on 2026-09-07:
- moonshotai/kimi-k2.5 present, 262144 context, expiration_date=null;
  input $0.45/M, output $2.25/M, cached input $0.07/M.
- moonshotai/kimi-k3 present, 1048576 context, expiration_date=null;
  input $3/M, output $15/M, cached input $0.30/M.
Both advertise structured_outputs and response_format. Listing is not a successful
inference/credit check. Refresh at startup; never infer availability from a web page.

Claude: official package claude-agent-sdk; verified model claude-sonnet-5.
Claude Code authentication documents setup-token for noninteractive personal usage.
Agent SDK quickstart/hosting describe API/cloud authentication and restrict offering
subscription login/limits to third-party product users without approval. Personal
OAuth must be explicit, never presented as unrestricted production entitlement.
No automatic API-key or paid-cloud fallback. SDK remains optional and off by default.

Render Free disk is ephemeral. Use Supabase REST checkpoints (free 500MB database)
for cross-deploy persistence, SQLite for local development or a mounted paid disk.
SQL setup is provided; provisioning an external account is a user setup step.
