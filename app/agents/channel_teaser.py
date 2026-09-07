"""Channel teaser agent: the public editorial hook under the original post."""

from __future__ import annotations

import logging

from app.models.atoms import ContentAtom
from app.models.content import ChannelTeaser, TeaserTimestamp
from app.models.transcript import Transcript
from app.utils.timecode import format_timecode

from .base import StructuredAgent
from .rendering import render_atoms

logger = logging.getLogger(__name__)

# A teaser built on invented timestamps is worse than one with none at all.
_MIN_TIMESTAMP_CONFIDENCE = 0.6


class ChannelTeaserAgent(StructuredAgent):
    prompt_file = "channel_teaser.md"
    name = "channel_teaser"

    async def write(self, transcript: Transcript, atoms: list[ContentAtom]) -> ChannelTeaser:
        reliable = [
            atom
            for atom in atoms
            if atom.confidence >= _MIN_TIMESTAMP_CONFIDENCE and atom.end_seconds > 0
        ]
        timestamps_allowed = self.settings.teaser_include_timestamps and bool(reliable)

        if self.settings.text_provider == "openrouter":
            text = await self.request_text(
                system=self.system_prompt(
                    timestamps_policy="Do not write timestamps; the application adds verified source times separately.",
                    output_instructions="Return only the teaser's 2–5 Russian sentences. No JSON, labels, explanation or timestamps.",
                ),
                user=f"Recording duration: {format_timecode(transcript.duration)}.\nCONTENT ATOMS:\n{render_atoms(atoms)}",
                temperature=0.75,
            )
            timestamps = (
                [
                    TeaserTimestamp(seconds=a.start_seconds, label=a.label[:120])
                    for a in sorted(reliable, key=lambda a: a.start_seconds)
                    if 0 <= a.start_seconds <= transcript.duration
                ][:5]
                if timestamps_allowed
                else []
            )
            return ChannelTeaser(teaser=text, timestamps=timestamps)

        system = self.system_prompt(
            timestamps_policy=(
                "Timestamps are reliable for this recording. Include 1-5 "
                "short thematic timestamp lines."
                if timestamps_allowed
                else "Timestamps are NOT reliable for this recording. Return an "
                "empty timestamps list."
            )
        )
        user = (
            f"Recording duration: {format_timecode(transcript.duration)}.\n"
            f"Detected language: {transcript.language or 'unknown'}.\n\n"
            "CONTENT ATOMS FOUND IN THIS RECORDING:\n"
            f"{render_atoms(atoms if self.settings.semantic_pipeline_enabled else atoms[:6])}"
        )
        teaser = await self.request(
            ChannelTeaser,
            system=system,
            user=user,
            temperature=0.75,
            max_tokens=(
                self.settings.groq_teaser_max_tokens
                if self.settings.text_provider == "groq"
                else self.settings.text_teaser_max_tokens
            ),
        )
        if not timestamps_allowed:
            teaser.timestamps = []
        else:
            limit = transcript.duration or float("inf")
            teaser.timestamps = [item for item in teaser.timestamps if 0 <= item.seconds <= limit][
                :5
            ]
        logger.info(
            "Teaser written (%s chars, %s timestamps)",
            len(teaser.teaser),
            len(teaser.timestamps),
        )
        return teaser
