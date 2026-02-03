# Draft Requirements (SAR) - Example

## REQ-SAR-IMG-001
- Statement: The system SHALL reconstruct a 2D SAR image with cross-range resolution <= 0.3 m.
- Type: performance
- Priority: must
- Verification: test
- Acceptance: Resolution <= 0.3 m for a 200m x 200m scene at slant range 5 km, BW >= 300 MHz.
- Notes: Use fixed seed=1234 and scenario=SAR_SCN_01.

## REQ-SAR-PROC-001
- Statement: The system SHALL reconstruct one image within <= 2.0 s on RTX 4080 with input size <= 360x432x119.
- Type: performance
- Priority: should
- Verification: test
- Acceptance: latency <= 2.0 s/image averaged over 10 runs, warm cache, GPU VRAM <= 20 GB.
