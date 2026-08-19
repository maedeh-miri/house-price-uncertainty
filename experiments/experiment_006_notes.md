# Experiment 006: Random Forest Hyperparameter Search

## Question

Can controlled Random Forest tuning improve the tree-based baseline
and make it competitive with the regularized linear models?

## Protocol

The experiment uses only the 1,758 observations in the frozen outer
training partition.

Evaluation uses the same deterministic 5-fold cross-validation
protocol used by the previous point-prediction experiments.

Within every fold, preprocessing is fitted only on the fold's fitting
rows. Validation rows are transformed without refitting.

Calibration and test partitions are not used.

The primary model-selection metric is OOF MAE. OOF RMSE is retained as
a secondary diagnostic metric.

## Search space

The predefined Random Forest search varied:

### Maximum depth

- `None`
- `16`

### Minimum samples per leaf

- `1`
- `2`
- `4`

### Maximum feature fraction

- `1.0`
- `0.7`

The following parameters were held fixed:

- `n_estimators = 500`
- `random_state = 2026`

This produced 12 candidate configurations.

## Selected Random Forest

The best configuration by OOF MAE was:

- `max_depth = None`
- `min_samples_leaf = 1`
- `max_features = 0.7`
- `n_estimators = 500`

Performance:

- OOF MAE: $16,173.00
- OOF RMSE: $27,366.95

## Improvement over the untuned baseline

The initial Random Forest baseline achieved:

- OOF MAE: $16,549.29
- OOF RMSE: $28,042.93

Controlled tuning therefore improved both point-prediction metrics.

The largest useful change in the selected configuration was reducing
`max_features` from 1.0 to 0.7.

## Comparison with regularized linear models

The selected ElasticNet model achieved:

- OOF MAE: $16,041.52
- OOF RMSE: $27,852.31

The selected Ridge model achieved:

- OOF MAE: $16,063.92
- OOF RMSE: $27,776.03

The tuned Random Forest does not achieve the lowest OOF MAE, but it
does achieve the lowest OOF RMSE among the evaluated point-prediction
models.

Because OOF MAE was defined as the primary model-selection metric
before model comparison, the selection criterion is not changed after
observing the stronger Random Forest RMSE.

## Interpretation

After controlled tuning, Random Forest becomes highly competitive with
the regularized linear models.

ElasticNet remains the MAE-optimal point model, while Random Forest
achieves the lowest OOF RMSE among the evaluated models. The lower
RMSE motivates later residual and tail-error analysis, but does not by
itself establish that Random Forest performs better specifically on
large individual errors.

Because the same cross-validation protocol is used for hyperparameter
selection and model comparison, these OOF results are treated as
development and model-selection estimates rather than unbiased final
performance estimates.

Final predictive performance will be assessed only after model
selection is complete, using the untouched evaluation partitions.

No calibration or test observations were used during tuning or model
selection.