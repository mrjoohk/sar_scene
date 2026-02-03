# Draft Requirements (Control) - Example

## REQ-CTRL-RT-001
- Statement: The controller SHALL run at 120 Hz with worst-case jitter <= 2 ms.
- Type: performance
- Priority: must
- Verification: test
- Acceptance: measured over 10 minutes, jitter <= 2 ms, missed cycles = 0.

## REQ-CTRL-STEP-001
- Statement: The longitudinal pitch attitude control SHALL achieve settling time <= 2.0 s and overshoot <= 10%.
- Type: performance
- Priority: should
- Verification: analysis
- Acceptance: step response evaluated on reference model; metrics computed with defined envelope.
