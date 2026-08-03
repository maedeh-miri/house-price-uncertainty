# Experiment: Lot Frontage Imputation Robustness

## Hypothesis

A hierarchical median based on `Neighborhood` and `Lot Config` should
outperform a global median, but a multivariate model may improve
reconstruction further.

## Evaluation protocols

1. **Repeated random cross-validation**
   - 5 folds
   - 10 repeats
   - 50 evaluations

2. **Conditional masking**
   - 20 repeated masks
   - masking probabilities are weighted by the observed missingness pattern
     of `Neighborhood` and `Lot Config`
   - the number of masked rows approximates the natural missingness rate

## Candidate strategies

- global median
- neighborhood median
- lot-configuration median
- hierarchical neighborhood-and-lot-configuration median
- model-based histogram gradient boosting

## Leakage controls

- every median is learned from the training split only
- the model-based candidate is fitted on the training split only
- `SalePrice` is not used as an imputation feature
- the naturally missing `Lot Frontage` values are not used as ground truth

## Important limitation

Conditional masking cannot validate neighborhoods or groups for which
all original frontage values are missing, because no ground-truth values
exist for those groups.

## Decision rule

Do not select the final production imputer from reconstruction error alone.
The leading candidates must later be compared inside the house-price
pipeline using price MAE/RMSE and prediction-interval coverage.
