# CAPMeeting

Civil Air Patrol squadron meeting agenda builder. Live at <https://capmeeting.freshskyai.com>.

CC enters meeting date + attendance + focus area → app produces a full 90-minute agenda with safety briefing, AE moment, drill objectives, character topic, and squadron business items, all calibrated to the inputs.

Flask app using the pinned `freshsky-common` package for the privacy-restricted LLM chain, security headers, and abuse limits. Meeting context must be de-identified: no names, rosters, CAP member IDs, contact details, street addresses, or sensitive operational information. Requests are stateless and agendas are not stored in an application database.

Generated agendas are drafts for human review. API responses are private/no-store and noindexed, likely identifiers are rejected before a provider call, and model output is validated against a narrow agenda schema.
