# Draft Requirements (SAR) - Example

## REQ-SAR-001
- Statement: The SAR processor SHALL achieve cross-range resolution <= 0.3 m.
- Type: performance
- Priority: must
- Verification: test
- Acceptance: cross-range resolution <= 0.3 m for scene=SAR_SCN_01, slant_range=12 km, BW=200 MHz, PRF=1200 Hz, seed=42.

## REQ-SAR-002
- Statement: The SAR processor SHALL achieve range resolution <= 0.5 m.
- Type: performance
- Priority: must
- Verification: test
- Acceptance: range resolution <= 0.5 m for scene=SAR_SCN_02, slant_range=10 km, BW=180 MHz, PRF=1500 Hz, seed=7.

## REQ-SAR-003
- Statement: The SAR pipeline SHALL complete processing latency <= 2.0 s/image.
- Type: performance
- Priority: must
- Verification: test
- Acceptance: latency <= 2.0 s/image for scene=SAR_SCN_03, input_shape=512x512, PRF=1100 Hz, seed=13.

## REQ-SAR-004
- Statement: The SAR backprojection unit SHALL keep VRAM usage <= 4.0 GB.
- Type: performance
- Priority: should
- Verification: analysis
- Acceptance: VRAM usage <= 4.0 GB for scene=SAR_SCN_04, input_shape=512x512, BW=250 MHz, seed=99.

## REQ-SAR-005
- Statement: The SAR processor SHALL achieve SNR >= 18 dB.
- Type: performance
- Priority: should
- Verification: test
- Acceptance: SNR >= 18 dB for scene=SAR_SCN_05, slant_range=9 km, PRF=1000 Hz, seed=21.
