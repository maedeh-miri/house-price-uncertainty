# Experiment 002: Median SalePrice Baseline

## Question

What level of prediction error is achieved by a feature-free median
baseline under the frozen training-only cross-validation protocol?

## Protocol

The experiment uses only the outer training partition from the frozen
primary random evaluation protocol.

The 1,758 training observations are evaluated with deterministic
5-fold cross-validation.

For each fold:

1. the median `SalePrice` is calculated from the fold's fitting rows;
2. that value is predicted for every validation observation;
3. MAE and RMSE are computed on the validation fold.

Calibration and test partitions are not used.

## Results

Overall out-of-fold performance:

- OOF MAE: $54,416.94
- OOF RMSE: $80,011.92

Fold-level summary:

- Mean fold MAE: $54,415.92
- Standard deviation of fold MAE: $2,690.37
- Mean fold RMSE: $79,776.75
- Standard deviation of fold RMSE: $6,788.95

Training medians across the five folds ranged from $159,000 to
$163,950.

## Interpretation

The median predictor establishes a deliberately weak but reproducible
reference point that uses no property features.

Subsequent feature-based models should improve meaningfully on both
the OOF MAE and OOF RMSE while using the same outer training partition
and cross-validation protocol.

The comparatively large RMSE reflects stronger sensitivity to large
prediction errors. Differences in fold-level RMSE will be examined
later through model error and residual analysis rather than by
changing the frozen cross-validation split.

## Limitations

This baseline does not use any property information and is not
intended as a competitive predictive model.

Its purpose is to define a transparent lower benchmark for subsequent
regularized linear and tree-based models.