# SAR Metrics Guide (Lite)

## Resolution
- Range resolution depends on bandwidth: delta_r ~ c/(2*BW)
- Cross-range (azimuth) resolution depends on aperture and geometry.
Acceptance examples:
- cross-range resolution <= 0.3 m under scenario=SAR_SCN_01.

## Performance
- latency (s/image), throughput (images/min), VRAM usage (GB)
- include nominal load and input size

## Quality
- SNR (dB), dynamic range (dB), PSLR/ISLR
- define measurement procedure and calibration conditions
