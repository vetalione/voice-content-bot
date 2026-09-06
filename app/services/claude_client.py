"""Optional personal OAuth Agent SDK adapter. No API-key/cloud fallback or tools."""

from __future__ import annotations

import asyncio
import os
import tempfile

from app.services.llm import LLMError, LLMGenerationError
from app.services.usage import capture_usage, current_usage


class ClaudeAgentClient:
    def __init__(self, settings):
        self.settings = settings
        self.cache_identity = ["claude-agent-sdk", settings.claude_model]

    async def aclose(self):
        pass

    async def chat_json(
        self, *, system, user, schema=None, schema_name="response", label="claude", **kwargs
    ):
        s = self.settings
        if not s.claude_enabled or not s.claude_model or not s.claude_code_oauth_token:
            raise LLMError(
                "Claude requires CLAUDE_ENABLED, CLAUDE_MODEL and CLAUDE_CODE_OAUTH_TOKEN; no API key fallback"
            )
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        except ImportError:
            raise LLMError(
                "Install optional requirements-claude.txt to enable the Claude Agent SDK"
            ) from None
        usage = current_usage.get()
        if usage:
            usage.claude_requests += 1
        # Remove competing credential routes from the child, without mutating the
        # web process environment or loading project/user hooks and MCP servers.
        clean = {
            k: ""
            for k in os.environ
            if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_"))
            or any(
                part in k for part in ("API_KEY", "OAUTH_TOKEN", "SERVICE_ROLE_KEY", "BOT_TOKEN")
            )
        }
        with tempfile.TemporaryDirectory(prefix="voice-claude-") as directory:
            options = ClaudeAgentOptions(
                model=s.claude_model,
                system_prompt=system,
                tools=[],
                allowed_tools=[],
                mcp_servers={},
                strict_mcp_config=True,
                permission_mode="dontAsk",
                setting_sources=[],
                cwd=directory,
                max_turns=2,
                max_budget_usd=s.claude_max_budget_usd,
                effort="medium",
                output_format={"type": "json_schema", "schema": schema} if schema else None,
                env={
                    **clean,
                    "CLAUDE_CODE_OAUTH_TOKEN": s.claude_code_oauth_token,
                    "CLAUDE_CONFIG_DIR": directory,
                    "CLAUDE_CODE_MAX_RETRIES": "1",
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(kwargs.get("max_tokens") or 8000),
                },
                stderr=lambda _: None,
            )
            actual_model = None
            try:
                async with asyncio.timeout(s.claude_timeout_seconds):
                    async for message in query(prompt=user, options=options):
                        actual_model = getattr(message, "model", None) or actual_model
                        if isinstance(message, ResultMessage):
                            meta = getattr(message, "usage", {}) or {}
                            # SDK total_cost_usd is a token-price estimate for OAuth,
                            # NOT an invoice or a deduction from the OpenRouter budget.
                            await capture_usage(
                                {
                                    "provider": "claude_agent_sdk",
                                    "stage": label,
                                    "configured_model": s.claude_model,
                                    "actual_model": actual_model,
                                    "input_tokens": meta.get("input_tokens"),
                                    "output_tokens": meta.get("output_tokens"),
                                    "reasoning_tokens": None,
                                    "reported_cost": None,
                                    "sdk_estimated_cost": getattr(message, "total_cost_usd", None),
                                    "turns": getattr(message, "num_turns", None),
                                }
                            )
                            if getattr(message, "is_error", False):
                                raise LLMError(f"Claude SDK result: {message.subtype}")
                            value = getattr(message, "structured_output", None)
                            if isinstance(value, dict):
                                return value
                            raise LLMGenerationError(
                                "Claude SDK did not return the requested structured object"
                            )
            except (LLMError, LLMGenerationError):
                raise
            except Exception as error:
                # SDK errors can contain subprocess environment/debug output.
                raise LLMError(
                    f"Claude SDK failed: {type(error).__name__}; no credential fallback"
                ) from None
        raise LLMGenerationError("Claude SDK ended without a result")
