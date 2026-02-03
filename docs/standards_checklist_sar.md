SAR Standards Checklist (Lite)

Requirements Checklist
- SHALL form used with numeric thresholds and units.
- Conditions include scene, seed, and key geometry parameters (slant range, BW, PRF).
- Verification method is test or analysis and is feasible.
- Acceptance criteria include measurable limits and reproducible setup.

Traceability Checklist (REQ -> UNIT -> CODE -> TEST -> CI -> Evidence)
- Each REQ links to a UNIT entry.
- Each REQ links to a CODE file path.
- Each REQ links to at least one TEST.
- CI run or workflow reference exists for tests.
- Evidence pack references trace matrix and artifacts.

Reproducibility Checklist
- Scenario ID and seed specified.
- Dataset or input_shape specified.
- Environment lock (Python version, deps) documented.
- Deterministic settings captured (seed, fixed config).
