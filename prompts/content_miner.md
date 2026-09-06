# Content Miner — extraction

Extract up to 6 distinct ideas or stories from this transcript window.
Return only JSON matching the schema. Each atom has exactly: title,
start_seconds, end_seconds, idea, type.
Use short Russian titles (about 5 words), ideas (one sentence, about 15 words),
and a short type such as idea, story, prediction or observation.
Use the absolute timestamps supplied; never invent facts. Skip filler.
Do not add scores, quotes, context or editorial metadata. An empty atoms list
is valid. Finish the complete JSON within the small output budget.

Publication exclusions:
{{voice_style}}
