# Experiment 007 Results: Symmetric Split-Conformal Calibration

## Status

Calibration completed under the pre-registered Experiment 007
protocol.

The primary test partition was not evaluated during this experiment.

## Frozen point model

The conformal calibration uses the previously selected point model:

- estimator: ElasticNet;
- alpha: `0.1`;
- l1 ratio: `0.9`;
- target: `SalePrice`;
- target scale: original dollars;
- target transformation: none.

The model specification was not changed after observing calibration
results.

## Data partitions used

- Training observations: 1,758
- Calibration observations: 586
- Test observations used: 0

The complete preprocessing and ElasticNet pipeline was fitted on the
training partition.

The fitted pipeline generated predictions for the calibration
partition without refitting.

## Conformal calibration result

The pre-registered uncertainty specification was:

- method: split conformal;
- nonconformity score: absolute residual;
- interval type: symmetric;
- nominal marginal coverage: 90%;
- finite-sample quantile rule:
  `ceil((n_calibration + 1) * coverage)`.

For 586 calibration observations:

`ceil((586 + 1) * 0.90) = 529`

The resulting conformal radius was:

`q_hat = 32616.33610273435`

Therefore, the frozen primary prediction interval is:

`[y_hat - 32616.33610273435, y_hat + 32616.33610273435]`

and its constant width is:

`65232.6722054687`

Dollar values may be rounded for presentation, but evaluation metrics
will use the full-precision values.

## Calibration score summary

Absolute calibration residuals had the following descriptive
statistics:

- minimum: `55.9556280215038`
- median: `10689.186753593764`
- mean: `15998.077212132399`
- maximum: `383777.3348426118`

These quantities are descriptive calibration diagnostics only.

They are not treated as independent estimates of final predictive
performance.

In particular, the calibration partition is not used to compare
alternative models or conformal methods.

## Interpretation

The conformal radius is substantially larger than the median
calibration absolute residual because the requested 90% interval is
determined by an upper order statistic of the calibration residual
distribution.

The very large maximum residual indicates that at least one
calibration observation has an unusually large point-prediction
error. No preprocessing, model, score, or interval-construction
decision is changed in response to this observation.

The symmetric method produces the same interval width for every
prediction and therefore does not adapt interval size to estimated
local difficulty or heteroscedasticity.

## Frozen result

The following values are now frozen for the primary random protocol:

- nominal coverage: `0.90`;
- conformal quantile rank: `529`;
- conformal radius: `32616.33610273435`;
- interval width: `65232.6722054687`;
- lower-bound clipping: disabled.

These values will be used unchanged for the final primary test
evaluation.

## Test-set status

The primary test partition has not been evaluated.

No test MAE, RMSE, coverage, subgroup coverage, or interval result was
observed during conformal calibration.

The next stage is the pre-specified final primary test evaluation.