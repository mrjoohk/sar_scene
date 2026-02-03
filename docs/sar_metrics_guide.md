# SAR Metrics Guide (Lite)

## Range Resolution
Acceptance criteria examples:
- Range resolution <= 0.5 m at scene=SAR_SCN_01, BW=200 MHz, PRF=1200 Hz.
- Range resolution <= 0.3 m at scene=SAR_SCN_02, BW=300 MHz, PRF=1500 Hz.
- Range resolution <= 0.6 m at scene=SAR_SCN_03, BW=160 MHz, PRF=1000 Hz.
- Range resolution <= 0.4 m at scene=SAR_SCN_04, BW=250 MHz, PRF=1400 Hz.
- Range resolution <= 0.7 m at scene=SAR_SCN_05, BW=140 MHz, PRF=900 Hz.

## Cross-range Resolution
Acceptance criteria examples:
- Cross-range resolution <= 0.3 m at scene=SAR_SCN_01, slant_range=12 km, PRF=1200 Hz.
- Cross-range resolution <= 0.4 m at scene=SAR_SCN_02, slant_range=10 km, PRF=1500 Hz.
- Cross-range resolution <= 0.5 m at scene=SAR_SCN_03, slant_range=8 km, PRF=1100 Hz.
- Cross-range resolution <= 0.35 m at scene=SAR_SCN_04, slant_range=15 km, PRF=1300 Hz.
- Cross-range resolution <= 0.45 m at scene=SAR_SCN_05, slant_range=9 km, PRF=1000 Hz.

## Bandwidth (BW)
Acceptance criteria examples:
- Effective BW >= 200 MHz for scene=SAR_SCN_01, seed=42.
- Effective BW within +/- 5% of 180 MHz for scene=SAR_SCN_02, seed=7.
- Effective BW >= 150 MHz for scene=SAR_SCN_03, seed=13.
- Effective BW within +/- 10% of 220 MHz for scene=SAR_SCN_04, seed=99.
- Effective BW >= 250 MHz for scene=SAR_SCN_05, seed=21.

## PRF
Acceptance criteria examples:
- PRF == 1200 Hz for scene=SAR_SCN_01, seed=42.
- PRF within +/- 2% of 1500 Hz for scene=SAR_SCN_02, seed=7.
- PRF >= 1000 Hz for scene=SAR_SCN_03, seed=13.
- PRF within +/- 5% of 1300 Hz for scene=SAR_SCN_04, seed=99.
- PRF == 900 Hz for scene=SAR_SCN_05, seed=21.

## Doppler
Acceptance criteria examples:
- Doppler centroid within +/- 5 Hz for scene=SAR_SCN_01, velocity=150 m/s.
- Doppler bandwidth >= 200 Hz for scene=SAR_SCN_02, velocity=180 m/s.
- Doppler centroid within +/- 10 Hz for scene=SAR_SCN_03, velocity=120 m/s.
- Doppler bandwidth >= 150 Hz for scene=SAR_SCN_04, velocity=160 m/s.
- Doppler centroid within +/- 8 Hz for scene=SAR_SCN_05, velocity=140 m/s.

## Platform Velocity
Acceptance criteria examples:
- Velocity = 150 m/s for scene=SAR_SCN_01, seed=42.
- Velocity within +/- 2 m/s of 180 m/s for scene=SAR_SCN_02, seed=7.
- Velocity >= 120 m/s for scene=SAR_SCN_03, seed=13.
- Velocity within +/- 3 m/s of 160 m/s for scene=SAR_SCN_04, seed=99.
- Velocity = 140 m/s for scene=SAR_SCN_05, seed=21.

## Latency (s/image)
Acceptance criteria examples:
- Latency <= 2.0 s/image for scene=SAR_SCN_01, input_shape=512x512, seed=42.
- Latency <= 1.5 s/image for scene=SAR_SCN_02, input_shape=256x256, seed=7.
- Latency <= 2.5 s/image for scene=SAR_SCN_03, input_shape=512x512, seed=13.
- Latency <= 3.0 s/image for scene=SAR_SCN_04, input_shape=1024x1024, seed=99.
- Latency <= 1.2 s/image for scene=SAR_SCN_05, input_shape=256x256, seed=21.

## VRAM Usage (GB)
Acceptance criteria examples:
- VRAM usage <= 4.0 GB for scene=SAR_SCN_01, input_shape=512x512, seed=42.
- VRAM usage <= 2.0 GB for scene=SAR_SCN_02, input_shape=256x256, seed=7.
- VRAM usage <= 6.0 GB for scene=SAR_SCN_03, input_shape=1024x1024, seed=13.
- VRAM usage <= 3.5 GB for scene=SAR_SCN_04, input_shape=512x512, seed=99.
- VRAM usage <= 2.5 GB for scene=SAR_SCN_05, input_shape=256x256, seed=21.

## SNR (dB)
Acceptance criteria examples:
- SNR >= 18 dB for scene=SAR_SCN_01, seed=42.
- SNR >= 20 dB for scene=SAR_SCN_02, seed=7.
- SNR >= 16 dB for scene=SAR_SCN_03, seed=13.
- SNR >= 22 dB for scene=SAR_SCN_04, seed=99.
- SNR >= 19 dB for scene=SAR_SCN_05, seed=21.

## PSLR (dB)
Acceptance criteria examples:
- PSLR <= -20 dB for scene=SAR_SCN_01, seed=42.
- PSLR <= -18 dB for scene=SAR_SCN_02, seed=7.
- PSLR <= -22 dB for scene=SAR_SCN_03, seed=13.
- PSLR <= -19 dB for scene=SAR_SCN_04, seed=99.
- PSLR <= -21 dB for scene=SAR_SCN_05, seed=21.

## ISLR (dB)
Acceptance criteria examples:
- ISLR <= -10 dB for scene=SAR_SCN_01, seed=42.
- ISLR <= -12 dB for scene=SAR_SCN_02, seed=7.
- ISLR <= -9 dB for scene=SAR_SCN_03, seed=13.
- ISLR <= -11 dB for scene=SAR_SCN_04, seed=99.
- ISLR <= -10.5 dB for scene=SAR_SCN_05, seed=21.
