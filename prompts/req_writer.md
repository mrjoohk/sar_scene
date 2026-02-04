# Domain Requirement Writer Guide (Aviation/Helicopter/SAR/Seeker/Control)

## Format (must)
- req_id: REQ-<AREA>-NNN (e.g., REQ-SAR-IMG-001)
- statement: The system SHALL ...
- type: functional | performance | interface | safety | security | testability
- priority: must | should | could
- verification_method: test | analysis | inspection | demonstration
- acceptance_criteria: quantified thresholds with units
- source_citations: KB citation keys (>=1)

## Rules
1) One requirement per req_id (no "and/or" bundling)
2) Use SHALL + measurable criteria (avoid ambiguous words: robust, fast, stable)
3) Acceptance criteria must include: metric + threshold + condition
4) Verification method must be feasible in CI or documented in evidence

## Domain metrics cheat sheet
### SAR Imaging
- Range resolution (m), Cross-range/Azimuth resolution (m)
- Bandwidth (Hz), center frequency (Hz), PRF (Hz), platform velocity (m/s)
- Latency per image (s), memory usage (GB), max scene size (m x m)
- SNR (dB), dynamic range (dB), PSLR/ISLR

### Flight/Control
- Settling time (s), overshoot (%), tracking error (deg/m), bandwidth (Hz)
- Stability margins (gain/phase margin), damping ratio
- Update rate (Hz), jitter (ms), worst-case latency (ms)

### Seeker / Missile
- Detection probability Pd, false alarm rate Pfa
- Track continuity (%), angle error (deg), range error (m)
- Update rate (Hz), end-to-end latency (ms)

## Good example
REQ-SAR-IMG-001:
The system SHALL reconstruct a 2D SAR image with cross-range resolution <= 0.3 m
for a 200 m x 200 m scene at slant range 5 km, using bandwidth >= 300 MHz.
Verification: test.
Acceptance: resolution <= 0.3 m measured on point-target impulse response
under seed=1234 scenario=SAR_SCN_01; latency <= 2.0 s/image on RTX 4080.
