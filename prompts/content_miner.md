# Content Miner — extraction

Extract up to 8 distinct ideas or stories from this transcript window.
Return only JSON matching the schema. Each atom has exactly: title,
start_seconds, end_seconds, idea, type.
Use short Russian titles (about 5 words), ideas (one sentence, about 15 words),
and a specific type: business_insight, marketing_observation, prediction, framework,
original_idea, strong_opinion, personal_story, funny_incident, travel_incident,
failure, relationship_observation, lesson, culture_observation or other.
Use the absolute timestamps supplied; never invent facts. Skip filler.
Do not add scores, quotes, context or editorial metadata. An empty atoms list
is valid. Finish the complete JSON within the small output budget.

Publication exclusions:
{{voice_style}}
