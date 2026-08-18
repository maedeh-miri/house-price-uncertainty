# Technical Report: House Price Prediction with Uncertainty

## Abstract

_To be completed after the final model and conformal experiments._

## 1. Introduction

A house-price estimate is more useful when accompanied by an honest
indication of uncertainty. This project studies point-prediction
quality, calibrated prediction intervals, subgroup reliability, and
performance under temporal distribution shift.

The primary prediction contract estimates expected residential market
value before the transaction is finalized.

The project is designed to answer not only whether a model can predict
house prices accurately, but also whether its uncertainty estimates
remain informative across different properties, neighborhoods, and
evaluation settings.

## 2. Dataset and provenance

The project uses the full Ames Housing dataset distributed with the
Ames Housing study.

The validated source contains:

- 2,930 residential property sales;
- 82 columns;
- no missing `SalePrice` values;
- 28 neighborhoods.

The raw dataset is not redistributed through this repository.

Download instructions, source information, loading semantics, and the
expected SHA256 checksum are documented in `data/README.md`.

Literal categorical values such as `NA` are preserved because they can
represent structural absence rather than unknown missingness.

A detailed dataset audit is available in:

`reports/data_audit.md`

## 3. Prediction contract and leakage controls

The primary model estimates expected residential market value before
the transaction is finalized.

The following columns are excluded from the primary predictor matrix:

- `Order`;
- `PID`;
- `Mo Sold`;
- `Yr Sold`;
- `Sale Type`;
- `Sale Condition`;
- `SalePrice`.

`SalePrice` is the prediction target.

`Order` and `PID` are retained as metadata rather than predictive
features.

Transaction-context variables are excluded from the primary model
because their availability is not guaranteed at pre-sale prediction
time. This exclusion does not imply that the variables are inherently
invalid; they may be evaluated later through explicitly defined
sensitivity analyses.

The complete feature-availability and semantic-role decisions are
documented in:

- `reports/feature_availability.md`;
- `reports/feature_schema.csv`.

## 4. Missing-data analysis

The dataset contains both genuine missing values and categorical
values representing structural absence.

The project therefore distinguishes:

```text
structural absence
!=
unknown missingness
```

Examples include properties without garages, basements, pools, or
other features.

This distinction is preserved during loading and will be respected by
the preprocessing pipeline rather than applying a single global
missing-value rule.

Detailed findings are documented in:

`reports/data_audit.md`

## 5. Preliminary preprocessing experiment

A dedicated experiment investigated reconstruction strategies for
`Lot Frontage`, which contains substantial genuine missingness.

The experiment compared:

- global median imputation;
- `Lot Config` median imputation;
- `Neighborhood` median imputation;
- hierarchical median imputation;
- HistGradientBoosting-based reconstruction.

The model-based approach produced the strongest reconstruction results
in the preliminary benchmark, while hierarchical median imputation was
the strongest simple baseline.

These results are treated as preprocessing research rather than a final
production decision.

The experiment was performed before the final evaluation protocol was
frozen. Any learned production imputer must therefore be fitted using
training data only.

Final preprocessing choices must also be evaluated according to their
downstream effect on:

- `SalePrice` prediction;
- model stability;
- conformal interval coverage;
- interval width;
- pipeline complexity.

## 6. Evaluation protocol

The evaluation design is frozen before final model fitting.

### 6.1 Primary random protocol

The primary evaluation uses a deterministic, target-independent
60/20/20 split:

| Partition | Rows | Percentage |
|---|---:|---:|
| Train | 1,758 | 60% |
| Calibration | 586 | 20% |
| Test | 586 | 20% |

Partition membership is generated using stable `Order` identifiers and
fixed random states `42` and `43`.

The split does not require or inspect `SalePrice` and is independent of
input DataFrame row order.

### 6.2 Partition roles

The training partition is used for:

- preprocessing development;
- cross-validation;
- feature engineering;
- model-family comparison;
- hyperparameter selection;
- final point-model fitting.

The calibration partition is reserved for conformal calibration.

The test partition is reserved for final evaluation.

### 6.3 Temporal stress test

A secondary forward-looking protocol evaluates temporal distribution
shift:

| Partition | Sale years | Rows |
|---|---|---:|
| Train | 2006–2008 | 1,941 |
| Calibration | 2009 | 648 |
| Test | 2010 | 341 |

The temporal protocol is treated as a stress test. Empirical conformal
coverage will be reported, but a distribution-free coverage guarantee
is not assumed under temporal distribution shift.

The complete protocol is documented in:

`reports/evaluation_protocol.md`

Exact row membership is frozen in:

`reports/evaluation_split_manifest.csv`

## 7. Leakage-safe preprocessing

The leakage-safe preprocessing pipeline has been implemented and
validated with automated unit and integration tests.

The pipeline enforces the following principle:

```text
Fit preprocessing on training data only
→ transform calibration without refitting
→ transform test without refitting
```

The current implementation includes:

- preservation of structural-absence semantics such as literal `NA`
  categories;
- train-only imputation of genuine missing numeric values;
- hierarchical `Lot Frontage` imputation using training-set
  `Neighborhood`, `Lot Config`, and global medians;
- explicit handling of structural garage absence for `Garage Yr Blt`;
- one-hot encoding for categorical variables;
- support for categorical levels not observed during training through
  unknown-category-safe encoding;
- treatment of `MS SubClass` as a categorical feature rather than a
  continuous numeric variable;
- optional numeric scaling for model families that require it.

The initial pipeline uses one-hot encoding for text-based ordinal
variables rather than imposing numeric distances between ordinal
levels. Explicit ordinal encoding remains a candidate for later
controlled comparison.

The preprocessing implementation is model-agnostic. Scaling can be
enabled for regularized linear models and disabled for tree-based
estimators.

Automated tests verify that:

- preprocessing parameters are learned from training data only;
- calibration and test transformation does not refit learned
  parameters;
- unseen categorical levels do not cause transformation failures;
- train, calibration, and test outputs have compatible feature
  dimensions;
- transformed model matrices contain finite values.

The full automated test suite currently contains 48 passing tests.

## 8. Point-prediction models

_Planned._

The initial model sequence will include:

1. a simple median-prediction baseline;
2. regularized linear regression;
3. a tree-based model.

Candidate target transformations such as `log1p(SalePrice)` will be
treated as experimental choices rather than assumed defaults.

Model selection will occur inside the training partition.

## 9. Conformal prediction

_Planned._

After the point-prediction model is frozen, the calibration partition
will be used to estimate conformal nonconformity quantiles.

Final uncertainty evaluation will consider:

- empirical interval coverage;
- mean and median interval width;
- coverage by price range;
- coverage by neighborhood;
- behavior under temporal distribution shift.

## 10. Subgroup evaluation

Neighborhood-level interval performance will be interpreted according
to subgroup size.

| Test subgroup size | Interpretation |
|---:|---|
| `n >= 50` | Primary interpretation |
| `20 <= n < 50` | Exploratory interpretation with warning |
| `n < 20` | Raw count and coverage only |

Small groups will not be removed from evaluation.

Subgroup reports should include sample size, empirical coverage, and
interval width where appropriate.

## 11. Final experiments and results

_To be completed after preprocessing, model selection, and conformal
calibration._

No final predictive or uncertainty results are reported at the current
stage.

## 12. Error analysis

_To be completed after final model evaluation._

The analysis will examine where predictive error and interval behavior
are weakest rather than reporting aggregate metrics alone.

## 13. Limitations

Current known limitations include:

- heterogeneous meanings of missing values;
- rare and unseen categorical levels;
- relatively small calibration and subgroup sample sizes;
- possible temporal distribution shift;
- high-dimensional representations after categorical encoding;
- lack of a guarantee of subgroup-conditional conformal coverage.

Additional limitations will be documented after final experiments.

## 14. Conclusion

_To be completed after final evaluation._
