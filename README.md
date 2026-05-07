# CAPMeeting

Civil Air Patrol squadron meeting agenda builder. Live at <https://capmeeting.freshskyai.com>.

CC enters meeting date + attendance + focus area → app produces a full 90-minute agenda with safety briefing, AE moment, drill objectives, character topic, and squadron business items, all calibrated to the inputs.

Standalone Flask app, no `freshsky_common` dependency. LLM auto-fallback chain. No PII, no rosters, no member data.
