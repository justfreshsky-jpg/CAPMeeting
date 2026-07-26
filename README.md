# CAPMeeting

Unofficial, experimental Civil Air Patrol squadron meeting agenda builder. It is not affiliated with or endorsed by Civil Air Patrol or CAP NHQ. Live at <https://capmeeting.freshskyai.com>.

CC enters meeting date + attendance + focus area → app produces a full 90-minute agenda with safety briefing, AE moment, drill objectives, character topic, and squadron business items, all calibrated to the inputs.

Flask app using the pinned `freshsky-common` package for the privacy-restricted LLM chain, security headers, and abuse limits. Meeting context must be de-identified: no names, rosters, CAPIDs or CAP member IDs, contact details, street addresses, PHI or patient information, incident or case identifiers, or operational secrets. Requests are stateless and agendas are not stored in an application database.

Generated agendas are drafts for human review. API responses are private/no-store and noindexed, likely identifiers are rejected before a provider call, and model output is validated against a narrow agenda schema.

Access includes three previews, then Civic costs $14.99/month with up to 40 usage units per day and 200 per month. Civic covers CivicOps only and does not unlock non-Civic products. Existing subscribers with an eligible broader entitlement remain supported.
