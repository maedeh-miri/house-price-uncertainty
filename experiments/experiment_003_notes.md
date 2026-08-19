# Experiment 003: Ridge Regression Baseline

## Question

How much does a leakage-safe regularized linear model improve over the
feature-free median baseline, and which predefined Ridge regularization
strength performs best under the frozen training-only cross-validation
protocol?

## Protocol

The experiment uses only the 1,758 observations in the outer training
partition from the frozen primary random evaluation protocol.

Evaluation uses the same deterministic 5-fold cross-validation
protocol used for the median baseline.

Within every fold, preprocessing is fitted only on the fold's fitting
rows. Validation rows are transformed without refitting preprocessing
parameters.

The Ridge pipeline includes numeric scaling because regularized linear
models are sensitive to feature scale.

Calibration and test partitions are not used.

## Initial Ridge baseline

The initial Ridge model used:

- `alpha = 1.0`
- no target transformation

Its out-of-fold performance was:

- OOF MAE: $16,441.44
- OOF RMSE: $27,633.28

## Controlled alpha search

The following Ridge alpha values were specified before comparing their
results:

- 0.01
- 0.1
- 1.0
- 10.0
- 100.0
- 1000.0

The primary selection metric was OOF MAE. OOF RMSE was retained as a
secondary diagnostic metric.

Results:

| Alpha | OOF MAE | OOF RMSE |
|---:|---:|---:|
| 10.0 | $16,063.92 | $27,776.03 |
| 1.0 | $16,441.44 | $27,633.28 |
| 100.0 | $16,539.42 | $28,663.03 |
| 0.1 | $16,734.88 | $27,875.87 |
| 0.01 | $16,782.56 | $27,900.72 |
| 1000.0 | $18,939.76 | $31,751.96 |

## Selection

`alpha = 10.0` is selected because it achieves the lowest OOF MAE,
which was defined as the primary model-selection metric before the
results were inspected.

Although `alpha = 1.0` achieves a slightly lower OOF RMSE, the
selection rule is not changed after observing the results.

## Comparison with the median baseline

The median baseline achieved:

- OOF MAE: $54,416.94
- OOF RMSE: $80,011.92

The selected Ridge model therefore reduces OOF MAE by approximately
70.5% and OOF RMSE by approximately 65.3% relative to the median
baseline.

This large improvement shows that the predictor set contains strong
signal beyond a feature-free central-value prediction.

## Interpretation

Most of the performance improvement comes from moving from the median
baseline to a feature-based regularized linear model.

Changing Ridge alpha from 1.0 to the selected value of 10.0 produces a
much smaller additional improvement, which suggests that reasonable
regularization is useful but is not the primary source of predictive
performance.

The difference between MAE-optimal and RMSE-optimal alpha values is
retained as a diagnostic observation rather than used to change the
predefined model-selection criterion.