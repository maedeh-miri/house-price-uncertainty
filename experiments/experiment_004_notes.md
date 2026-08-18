# Experiment 004: ElasticNet Baseline

## Question

Can a mixed L1/L2 regularized linear model improve on the selected
Ridge baseline under the frozen training-only cross-validation
protocol?

## Protocol

The experiment uses only the 1,758 observations in the frozen outer
training partition.

Evaluation uses the same deterministic 5-fold cross-validation
protocol used for the median and Ridge baselines.

Within every fold, preprocessing is fitted only on the fold's fitting
rows. Validation rows are transformed without refitting.

Calibration and test partitions are not used.

The target remains on its original scale.

## Hyperparameter search

The predefined ElasticNet search space was:

### Alpha

- 0.001
- 0.01
- 0.1
- 1.0
- 10.0
- 100.0
- 1000.0

### L1 ratio

- 0.1
- 0.5
- 0.9

This produces 21 candidate configurations.

The primary selection metric was OOF MAE. OOF RMSE was retained as a
secondary diagnostic metric.

## Selected ElasticNet configuration

The best configuration by OOF MAE was:

- `alpha = 0.1`
- `l1_ratio = 0.9`

Performance:

- OOF MAE: $16,041.52
- OOF RMSE: $27,852.31
- convergence warnings: 0

All evaluated candidates completed without convergence warnings.

## Comparison with Ridge

The selected Ridge model achieved:

- OOF MAE: $16,063.92
- OOF RMSE: $27,776.03

ElasticNet improves OOF MAE by approximately $22.40, or 0.14%,
relative to Ridge.

Ridge, however, achieves an OOF RMSE approximately $76.28 lower than
ElasticNet.

The two regularized linear models are therefore near-tied in predictive
performance under the frozen cross-validation protocol.

ElasticNet is retained as the MAE-optimal regularized-linear
configuration, while Ridge remains a simpler and highly competitive
reference model.

## Comparison with the median baseline

The median baseline achieved:

- OOF MAE: $54,416.94
- OOF RMSE: $80,011.92

The selected ElasticNet model reduces OOF MAE by approximately 70.5%
and OOF RMSE by approximately 65.2% relative to the feature-free
median baseline.

## Interpretation

Both Ridge and ElasticNet capture substantial predictive signal from
the feature set.

The very small difference between their cross-validated errors
suggests that the choice between these two regularized linear model
families is much less important than the transition from a
feature-free baseline to a feature-based model.

No additional hyperparameter search is introduced after observing
these results.