# Experiment 010 — Conformal Sensitivity Analysis Results

## Summary

Experiment 010 evaluates the sensitivity of the conformal uncertainty
system under different nominal coverage targets.

The analysis keeps the primary predictive system frozen and changes
only the requested conformal coverage level.

Evaluated configurations:

- 80% nominal coverage
- 90% nominal coverage
- 95% nominal coverage

---

## Frozen System

Point model:

- Model: ElasticNet
- alpha: 0.1
- l1_ratio: 0.9

Conformal method:

- Method: split conformal prediction
- Score: absolute residual
- Interval type: symmetric

The primary 90% configuration remains the main uncertainty result.

---

## Results

| Nominal Coverage | Empirical Coverage | Mean Interval Width |
|---|---:|---:|
| 80% | 78.67% | $43,985.81 |
| 90% | 91.47% | $65,232.67 |
| 95% | 95.90% | $89,303.16 |

---

## Interpretation

Increasing the nominal coverage target increases the prediction
interval width.

The 80% configuration produces narrower intervals but accepts a higher
probability of missed observations.

The 95% configuration achieves the highest empirical coverage but
requires substantially wider intervals.

The 90% configuration provides a balance between reliability and
interval sharpness.

---

## Primary Selection

The 90% conformal configuration remains the primary uncertainty result.

It achieves:

- empirical coverage above the target level;
- substantially narrower intervals than the 95% configuration;
- stronger reliability than the 80% configuration.

The sensitivity analysis supports the original choice of 90% coverage
as a practical operating point.

---

## Conclusion

Experiment 010 demonstrates the expected coverage-width trade-off of
the conformal prediction system.

Higher uncertainty guarantees require wider intervals, while narrower
intervals provide less coverage.

The frozen 90% conformal setting is retained as the headline uncertainty
configuration.