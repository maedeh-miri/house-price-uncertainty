# Experiment 005: Random Forest Baseline

## Question

Does a nonlinear tree-based ensemble improve on the regularized linear
baselines under the frozen training-only cross-validation protocol?

## Protocol

The experiment uses only the 1,758 observations in the frozen outer
training partition.

Evaluation uses the same deterministic 5-fold cross-validation
protocol used for the median, Ridge, and ElasticNet baselines.

Within every fold, preprocessing is fitted only on the fold's fitting
rows. Validation rows are transformed without refitting.

Calibration and test partitions are not used.

Numeric scaling is disabled because Random Forest does not require
feature standardization.

## Model

The baseline Random Forest configuration uses:

- 500 trees
- `random_state = 2026`
- all other estimator parameters at their default values

## Results

Out-of-fold performance:

- OOF MAE: $16,549.29
- OOF RMSE: $28,042.93

Fold-level summary:

- Mean fold MAE: $16,549.09
- Standard deviation of fold MAE: $1,396.71
- Mean fold RMSE: $27,730.52
- Standard deviation of fold RMSE: $4,655.28

## Comparison with regularized linear models

The selected Ridge model achieved:

- OOF MAE: $16,063.92
- OOF RMSE: $27,776.03

The selected ElasticNet model achieved:

- OOF MAE: $16,041.52
- OOF RMSE: $27,852.31

The Random Forest baseline therefore does not improve on either
regularized linear model under the frozen cross-validation protocol.

## Interpretation

The nonlinear Random Forest model substantially outperforms the
feature-free median baseline, confirming that it captures useful
predictive structure.

However, its performance is slightly weaker than both Ridge and
ElasticNet. This indicates that nonlinear tree-based modeling does not
automatically provide an advantage over the regularized linear
representation used in this project.

The result is retained rather than changing the evaluation split or
model-selection criteria after observing performance.

Additional tree-based tuning, if performed, must remain restricted to
the outer training partition.