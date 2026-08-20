# Experiment 008 — Post-hoc Primary Test Diagnostics Results

## Status

Completed.

This experiment is post-hoc and descriptive. The frozen primary model,
preprocessing pipeline, conformal radius, nominal coverage target, and
evaluation split were not changed.

## 1. Reproduction check

The diagnostic pipeline exactly reproduced the frozen primary test
evaluation:

- Test rows: 586
- MAE: $15,957.31
- RMSE: $33,434.09
- Empirical 90% coverage: 91.47%
- Covered observations: 536 / 586
- Mean interval width: $65,232.67

This confirms that the diagnostic analysis used the same frozen primary
system as the final evaluation.

## 2. Residual distribution

The residual distribution was centered close to zero:

- Mean residual: $81.97
- Median residual: $516.39
- Median absolute error: $10,730.88
- 90th percentile absolute error: $31,704.04
- 95th percentile absolute error: $41,391.12
- 99th percentile absolute error: $89,637.30
- Maximum absolute error: $594,968.46

The large gap between the median absolute error and the maximum error
shows that the test error distribution contains a pronounced upper tail.

## 3. RMSE tail sensitivity

The frozen full-test RMSE was:

- $33,434.09

Diagnostic RMSE after removing the largest absolute-error observations:

- largest 1% removed: $18,981.31
- largest 5% removed: $15,476.23
- largest 10% removed: $13,580.12

These values are diagnostic only and do not replace the frozen final
RMSE.

The results show that a relatively small set of extreme prediction
errors has a large influence on squared-error performance.

## 4. Largest observed error

The largest absolute error occurred for:

- Order: 1499
- Neighborhood: Edwards
- True SalePrice: $160,000
- Prediction: $754,968.46
- Residual: -$594,968.46
- Absolute error: $594,968.46
- Conformal result: missed below the lower interval bound

This observation strongly influences overall RMSE.

No model modification is made in response to this case.

## 5. Underprediction versus overprediction

The direction of point-prediction errors was nearly balanced:

- Underpredictions: 301 / 586 (51.37%)
- Overpredictions: 285 / 586 (48.63%)

However, their squared-error behavior differed:

- Underprediction RMSE: $25,273.00
- Overprediction RMSE: $40,297.02

The larger overprediction RMSE is strongly affected by the extreme
Order 1499 observation.

The overall mean residual remained close to zero, so the full test set
does not show a large global directional bias.

## 6. Training-derived price-band diagnostics

Price bands were defined using quartiles from the frozen training target:

- Q1: SalePrice <= $129,425
- Q2: $129,425 < SalePrice <= $160,100
- Q3: $160,100 < SalePrice <= $210,000
- Q4: SalePrice > $210,000

Results:

| Band | n | MAE | RMSE | Coverage |
|---|---:|---:|---:|---:|
| Q1 | 154 | $13,819.98 | $18,373.91 | 94.16% |
| Q2 | 149 | $14,498.89 | $50,720.49 | 95.97% |
| Q3 | 126 | $12,622.72 | $17,494.06 | 93.65% |
| Q4 | 157 | $22,114.07 | $33,971.60 | 82.80% |

The Q2 RMSE is heavily influenced by the extreme Order 1499 error and
should not be interpreted as evidence of generally poor performance in
that price band.

The highest-price band, Q4, shows both higher MAE and lower empirical
coverage than the lower three bands.

This is a descriptive post-hoc finding, not a basis for changing the
primary model or conformal radius.

## 7. Conformal misses

The primary test contained 50 interval misses:

- Above upper bound: 26
- Below lower bound: 24

Misses were therefore approximately balanced in direction.

Distribution across training-derived price bands:

- Q1: 9 misses
- Q2: 6 misses
- Q3: 8 misses
- Q4: 27 misses

More than half of all misses occurred in the highest-price band.

Error magnitude differed substantially between covered and missed
observations:

- Covered-observation MAE: $11,264.41
- Covered-observation RMSE: $14,091.16
- Missed-observation MAE: $66,265.18
- Missed-observation RMSE: $104,749.68

## 8. Neighborhood diagnostics

Two Neighborhoods met the pre-specified threshold for primary
interpretation:

### CollgCr

- n: 52
- MAE: $10,209.50
- RMSE: $14,364.70
- Coverage: 96.15%

### NAmes

- n: 81
- MAE: $11,878.04
- RMSE: $16,509.82
- Coverage: 97.53%

Both achieved empirical coverage above the 90% nominal target.

An important exploratory finding was observed for NridgHt:

- n: 37
- MAE: $31,985.90
- RMSE: $48,938.56
- Coverage: 67.57%

Because `20 <= n < 50`, this result remains exploratory under the
frozen subgroup policy.

Five of the ten largest absolute errors occurred in NridgHt.

## 9. Main interpretation

The primary test MAE remained close to the development OOF estimate,
while RMSE increased substantially.

The post-hoc diagnostics show that this difference is largely explained
by a small upper tail of very large prediction errors rather than by a
broad deterioration across most test observations.

The highest-price properties also showed weaker empirical conformal
coverage and larger absolute errors.

These findings motivate discussion of tail behavior, subgroup
heterogeneity, and the limitations of constant-width symmetric
conformal intervals.

They do not change the frozen primary result.

## 10. Limitations

This analysis is post-hoc.

It does not establish causal or mechanistic explanations for individual
errors.

Neighborhood and price-band results are descriptive and may be unstable
for smaller sample sizes.

The current split-conformal method targets marginal coverage rather than
equal conditional coverage across price bands or neighborhoods.

## 11. Next step

The next planned evaluation is the pre-defined temporal stress test:

- Train: 2006–2008
- Calibration: 2009
- Test: 2010

The primary random-split model specification and 90% conformal method
remain unchanged.