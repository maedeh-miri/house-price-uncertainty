# Experiment 007: Symmetric Split-Conformal Calibration

## Status

Pre-registered before conformal calibration.

No calibration residuals, conformal radius, or test-set performance
were inspected when this protocol was written.

## Research question

Can the selected point-prediction pipeline be combined with
split-conformal calibration to produce useful 90% prediction intervals
for Ames Housing prices while keeping model development, uncertainty
calibration, and final evaluation strictly separated?

## Point model

The point-prediction model was selected before conformal calibration.

The frozen model specification is:

- estimator: ElasticNet;
- alpha: `0.1`;
- l1 ratio: `0.9`;
- target: `SalePrice`;
- target scale: original dollar scale;
- target transformation: none.

ElasticNet was selected because OOF MAE was defined as the primary
model-selection metric before comparing the candidate point models.

The point-model decision will not be reopened using calibration
results.

## Primary evaluation protocol

The primary target-independent random protocol contains:

- Training: 1,758 observations;
- Calibration: 586 observations;
- Test: 586 observations.

The complete point-prediction pipeline, including all learned
preprocessing transformations, will be fitted on the complete training
partition only.

The fitted pipeline will then generate predictions for the calibration
partition without refitting preprocessing or model parameters.

The test partition is not used during this experiment.

## Conformal method

The primary uncertainty method is symmetric split conformal prediction.

For each calibration observation, the nonconformity score is the
absolute residual:

`score_i = abs(y_i - y_hat_i)`

The nominal marginal coverage level is fixed at:

`0.90`

The finite-sample conformal order-statistic rank is:

`ceil((n_calibration + 1) * coverage)`

With 586 calibration observations:

`ceil((586 + 1) * 0.90) = 529`

The conformal radius is therefore the 529th smallest absolute
calibration residual.

For a future point prediction `y_hat`, the prediction interval is:

`[y_hat - q_hat, y_hat + q_hat]`

where `q_hat` is the calibrated conformal radius.

## Interval boundary policy

Intervals are evaluated as closed intervals:

`lower <= y_true <= upper`

The primary method does not clip negative lower bounds to zero.

Any rounding applied in reports or visualizations is display-only.
Coverage and point-prediction metrics will be computed using the
unrounded predictions and interval bounds.

## Why symmetric absolute-residual conformal?

This method is intentionally used as the primary uncertainty baseline
because it is:

- simple;
- model-agnostic;
- auditable;
- reproducible;
- compatible with the frozen train/calibration/test design.

Under the standard exchangeability assumptions, split conformal
targets finite-sample marginal coverage.

The method does not guarantee 90% conditional coverage for every
neighborhood or price range.

## Known limitation

Symmetric absolute-residual conformal produces a single calibrated
radius.

Consequently, every prediction interval has the same width:

`interval width = 2 * q_hat`

This baseline therefore does not explicitly model heteroscedasticity.

A wider or narrower interval cannot adapt to an individual property's
predicted difficulty.

This limitation will be reported rather than addressed by selecting a
different conformal method after inspecting calibration or test
performance.

## Calibration-set restrictions

The calibration partition may be used only to:

- generate predictions from the already fitted point model;
- compute absolute residual nonconformity scores;
- compute the predefined finite-sample conformal quantile.

Calibration results must not be used to:

- change preprocessing;
- select another point model;
- retune ElasticNet;
- change the nominal coverage;
- choose another score function;
- select between symmetric and asymmetric intervals;
- choose a different random split.

Calibration-set coverage is not treated as an independent estimate of
generalization performance because the same observations determine
the conformal radius.

## Frozen final test metrics

Before observing the test targets, the following primary final metrics
are fixed.

### Point prediction

- MAE;
- RMSE.

### Prediction intervals

- empirical coverage;
- number covered / total observations;
- interval width.

Because the primary symmetric conformal method has a constant radius,
all primary test intervals have the same width `2 * q_hat`.

## Neighborhood subgroup diagnostics

Neighborhood-level interval performance will report:

- subgroup sample size;
- number of covered observations;
- empirical subgroup coverage.

Interpretation follows the previously frozen policy:

- `n >= 50`: primary subgroup interpretation;
- `20 <= n < 50`: exploratory interpretation with a small-sample
  warning;
- `n < 20`: descriptive count and raw coverage only.

Subgroup results are diagnostic.

They do not imply subgroup-conditional conformal guarantees.

## Test-set restrictions

The primary test partition remains untouched during conformal
calibration.

Test results will not be used to:

- change the point model;
- change preprocessing;
- alter the conformal score;
- change the coverage level;
- change the interval construction;
- change random seeds or split membership.

The final primary test evaluation will occur only after calibration is
complete and its result has been frozen.

## Temporal stress test

After the primary random protocol is completed, the same point-model
and conformal specifications will be applied to the frozen temporal
protocol:

- Training: 2006-2008;
- Calibration: 2009;
- Test: 2010.

Because the temporal protocol intentionally introduces possible
distribution shift, its interval coverage will be reported as
empirical temporal coverage rather than presented as an unconditional
distribution-free guarantee.

## Pre-registered interpretation policy

The symmetric conformal method will not be declared successful solely
because its empirical test coverage exceeds 90%.

Coverage must be interpreted together with interval width.

Likewise, coverage below 90% will be reported as an empirical result,
not used as justification for modifying the already evaluated test
protocol.

No conformal method will be selected post hoc using primary test
performance.