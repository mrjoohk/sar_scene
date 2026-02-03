# Requirements Baseline: BL-SAR-DEMO-001

## REQ-SAR-IMG-010
- **Statement**: The SAR imaging system SHALL achieve cross-range resolution <= 0.3 m under defined operating conditions.
- **Type**: performance
- **Priority**: must
- **Verification**: test
- **Acceptance**: Cross-range resolution <= 0.3 m for scene=SAR_SCN_01, slant_range=12 km, BW=200 MHz, PRF=1200 Hz, seed=42.
- **Citations**: KB:33d2e98d7555ef33#c0

## REQ-SAR-PROC-011
- **Statement**: The SAR pipeline SHALL process each image with latency <= 2.0 s/image and VRAM usage <= 4.0 GB under nominal load.
- **Type**: performance
- **Priority**: must
- **Verification**: test
- **Acceptance**: Latency <= 2.0 s/image and VRAM usage <= 4.0 GB for scenario=SAR_SCN_03, seed=13, input_shape=512x512, hw_profile=cpu.
- **Citations**: KB:9e8a7973f322749b#c2

## REQ-SAR-QUAL-012
- **Statement**: The SAR processor SHALL achieve image quality thresholds for SNR, PSLR, and ISLR.
- **Type**: performance
- **Priority**: must
- **Verification**: test
- **Acceptance**: SNR >= 18 dB, PSLR <= -20 dB, and ISLR <= -10 dB for scene=SAR_SCN_01, slant_range=12 km, PRF=1200 Hz, seed=42.
- **Citations**: KB:9e8a7973f322749b#c2
