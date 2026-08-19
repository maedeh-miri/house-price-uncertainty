# Experiment 008 — Post-hoc Primary Test Diagnostics

## Status

Analysis plan defined after completion of the frozen primary test
evaluation and before detailed residual inspection.

This experiment is explicitly post-hoc.

Its purpose is to understand the behavior of the already frozen
primary system. It must not be used to select a new model,
hyperparameter setting, preprocessing rule, conformal radius, split,
or primary evaluation metric.

## 1. Motivation

The frozen primary test evaluation produced:

- Test rows: 586
- MAE: $15,957.31
- RMSE: $33,434.09
- Nominal conformal coverage: 90%
- Empirical conformal coverage: 91.47%
- Covered observations: 536 / 586
- Mean interval width: $65,232.67

Development OOF performance for the selected ElasticNet model was:

- OOF MAE: $16,041.52
- OOF RMSE: $27,852.31

The final test MAE is close to the development estimate, while test
RMSE is substantially larger.

This suggests that a relatively small number of large errors may have
a stronger influence on squared-error performance, but this
interpretation has not yet been verified.

Experiment 008 investigates that behavior without modifying the
frozen predictive system.

## 2. Frozen predictive system

The diagnostic analysis must reproduce the exact primary system.

### Point model

- Model: ElasticNet
- alpha: 0.1
- l1_ratio: 0.9
- target: raw `SalePrice`
- target transformation: none
- training partition: frozen primary Train
- training rows: 1,758

### Conformal system

- method: split conformal
- score: absolute residual
- nominal coverage: 0.90
- calibration rows: 586
- finite-sample quantile rank: 529
- conformal radius: `32616.33610273435`
- interval type: symmetric
- lower-bound clipping: disabled

### Test partition

- rows: 586
- already evaluated once
- considered consumed for model-development purposes

No model or uncertainty parameter may be changed in response to this
analysis.

## 3. Residual definitions

For each primary-test observation:

```text
residual = y_true - y_pred
absolute_error = abs(residual)
squared_error = residual ** 2
```

Residual sign convention:

```text
residual > 0
→ model underpredicted the sale price

residual < 0
→ model overpredicted the sale price
```

Conformal coverage is defined using closed interval endpoints:

```text
lower <= y_true <= upper
```

A missed observation will be classified as either:

```text
above_upper
```

or:

```text
below_lower
```

## 4. Planned diagnostic analyses

### 4.1 Overall residual distribution

Report:

- mean residual;
- median residual;
- residual standard deviation;
- minimum residual;
- maximum residual;
- mean absolute error;
- root mean squared error.

For absolute errors, report:

- median;
- 75th percentile;
- 90th percentile;
- 95th percentile;
- 99th percentile;
- maximum.

These summaries will be used to determine whether the difference
between MAE and RMSE is consistent with a long upper tail of large
prediction errors.

### 4.2 Largest prediction errors

Rank primary-test observations by absolute error.

Inspect the fixed top 10 observations.

For each top-error observation report:

- stable `Order` identifier;
- Neighborhood;
- true `SalePrice`;
- point prediction;
- signed residual;
- absolute error;
- conformal lower bound;
- conformal upper bound;
- whether the observation was covered.

The top-10 threshold is fixed before detailed inspection and must not
be changed after seeing the observations.

### 4.3 RMSE tail sensitivity

To quantify how strongly large residuals influence RMSE, recompute
diagnostic RMSE after removing the observations with the largest
absolute errors.

Report:

- full test RMSE;
- RMSE excluding the largest 1% of absolute errors;
- RMSE excluding the largest 5%;
- RMSE excluding the largest 10%.

These are diagnostic quantities only.

They are not alternative final-performance metrics and must not
replace the frozen full-test RMSE.

### 4.4 Underprediction versus overprediction

Separate observations according to residual sign.

For each direction report:

- sample count;
- percentage of test observations;
- mean signed residual;
- mean absolute error;
- RMSE.

This analysis tests whether large errors are concentrated primarily
in underprediction or overprediction.

### 4.5 Error versus property price

Price-band boundaries will be derived from the frozen training target,
not selected from primary-test performance.

Using training-target quartiles, assign test observations to four
price bands:

- Q1;
- Q2;
- Q3;
- Q4.

For each band report:

- test sample size;
- mean true `SalePrice`;
- MAE;
- RMSE;
- mean signed residual;
- conformal empirical coverage.

This analysis evaluates whether prediction error increases for more
expensive properties.

No price-band-specific model adjustment will be performed.

### 4.6 Conformal misses

The frozen primary test contains 50 observations outside the
prediction interval.

Report:

- total misses;
- misses above the upper bound;
- misses below the lower bound;
- mean absolute error among covered observations;
- mean absolute error among missed observations;
- RMSE among covered observations;
- RMSE among missed observations.

Also report the distribution of misses across training-derived price
bands.

This analysis is descriptive and does not change the frozen
conformal radius.

### 4.7 Neighborhood diagnostics

Use the existing frozen subgroup interpretation policy:

```text
n >= 50
→ primary interpretation

20 <= n < 50
→ exploratory interpretation

n < 20
→ descriptive only
```

For each Neighborhood report:

- n;
- MAE;
- RMSE;
- mean signed residual;
- covered count;
- empirical coverage;
- interpretation tier.

No neighborhood-specific conformal method or model will be selected
from these results.

## 5. Outputs

The experiment should produce:

```text
experiments/results/posthoc_diagnostics_summary.json
```

The JSON should contain machine-readable overall, tail, price-band,
conformal-miss, and Neighborhood summaries.

A compact top-10 error table may also be stored as:

```text
experiments/results/posthoc_top_errors.csv
```

The project will not commit a full row-level file containing all 586
test targets and predictions unless a later reproducibility need
clearly justifies it.

## 6. Interpretation rules

The analysis is intended to answer questions such as:

- Is the high test RMSE driven by a relatively small error tail?
- Are the largest errors mostly underpredictions or overpredictions?
- Do prediction errors increase with property price?
- Are conformal misses concentrated in particular price regions?
- Do some Neighborhoods show notably different empirical behavior?

The analysis must distinguish observation from explanation.

For example:

Acceptable:

> Large absolute residuals are concentrated among higher-priced
> properties.

Not yet acceptable without additional evidence:

> The model fails on expensive houses because ElasticNet cannot model
> nonlinear luxury-market effects.

The second statement proposes a causal or mechanistic explanation
that would require additional analysis.

## 7. Non-negotiable safeguards

Experiment 008 must not be used to:

- change the primary ElasticNet model;
- change alpha or l1_ratio;
- select another previously evaluated model;
- change preprocessing;
- change the primary random split;
- change the conformal radius;
- change nominal primary coverage;
- remove difficult test observations;
- replace the frozen test MAE or RMSE;
- redefine the primary result based on post-hoc findings.

Any future modeling motivated by these diagnostics must be treated as
a new exploratory or follow-up experiment and must not overwrite the
frozen primary evaluation.

## 8. Relationship to later experiments

After Experiment 008 is completed and documented, the intended
sequence is:

```text
Post-hoc primary diagnostics
→ Temporal stress test
→ Secondary 80% / 90% / 95% conformal sensitivity analysis
→ Technical report
→ Release preparation
```

The primary 90% conformal result remains the headline uncertainty
result regardless of later sensitivity analyses.
