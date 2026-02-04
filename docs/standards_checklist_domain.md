# Standards & Checklist (Domain) - Lite

This document is an internal checklist to ensure requirements and tests are consistent with aerospace/defense development practices.

## Requirements checklist
- Each requirement uses SHALL and is singular.
- Acceptance criteria is measurable (numbers + units).
- Conditions are defined (scene size, slant range, bandwidth, hardware, seed, scenario).
- Verification method is specified (test/analysis/inspection/demonstration).
- At least one source citation exists.

## Traceability checklist
- REQ -> UNIT -> CODE -> TEST links exist.
- CI run produces JUnit and Coverage XML.
- Evidence pack includes baseline_id, commit, CI run URL, artifacts, checksum (sha256).

## Governance / Reproducibility checklist (LLM/RL)
- Dataset version + checksum recorded.
- Scenario version + checksum recorded.
- Seed recorded.
- Environment lock recorded (requirements.txt hash or container digest).
