SAR Requirements Writer (Lite)

Purpose
- Provide concise, testable SAR requirements for automated validation.

Rules
- Use SHALL statements.
- Include numeric thresholds with units.
- State conditions: scene, slant range, bandwidth (BW), PRF, platform velocity if relevant.
- Specify a verification method: test or analysis.

Good Examples
- The SAR processor SHALL achieve cross-range resolution <= 0.3 m for scene S1 at slant range 12 km, BW 200 MHz, PRF 1200 Hz, verified by test.
- The SAR pipeline SHALL produce latency <= 2.0 s/image under nominal load for scene S2 with seed 42, verified by test.
- The backprojection unit SHALL keep VRAM usage <= 4.0 GB for input_shape 512x512 and BW 150 MHz, verified by analysis.

Bad Examples
- The SAR processor should be fast. (no SHALL, no numeric threshold)
- Provide good resolution for most scenes. (ambiguous, no conditions)
- Latency is low. (no unit, no threshold, no verification)
