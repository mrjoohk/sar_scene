# Requirements Writer Policy (defense-lite)

Write requirements in the following format:

- req_id: REQ-<DOMAIN>-<AREA>-NNN
- statement: The system SHALL ...
- type: functional | performance | interface | safety | security | testability
- priority: must | should | could
- verification_method: test | analysis | inspection | demonstration
- acceptance_criteria: quantified thresholds (must include numbers/units)
- source_citations: list of citation keys from KB search (at least 1)

Rules:
- No ambiguous words (fast, robust, stable, etc.) unless quantified.
- One requirement per req_id (do not mix multiple independent requirements).
- Must be verifiable.
