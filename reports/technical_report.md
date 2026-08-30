# Technical Report: House Price Prediction with Uncertainty

## Abstract

This project develops and evaluates a reproducible machine-learning workflow for residential sale-price prediction with calibrated uncertainty estimates.

The study uses the Ames Housing dataset and places particular emphasis on leakage prevention, explicit evaluation roles, pre-specified model selection, split conformal prediction, subgroup reliability, and robustness under temporal distribution shift.

A deterministic 60/20/20 primary split separates model development, conformal calibration, and final testing. Model selection is performed using five-fold cross-validation on the training partition only. Among the evaluated point-prediction models, ElasticNet achieved the lowest pre-specified cross-validated MAE and was frozen as the primary point model before the final test evaluation.

On the held-out primary test set, the final ElasticNet model achieved an MAE of $15,957.31 and an RMSE of $33,434.09. A pre-registered 90% split-conformal interval achieved 91.47% empirical coverage, with a mean interval width of $65,232.67.

Post-hoc diagnostics showed that a relatively small number of large residuals strongly influenced RMSE. Secondary analyses examined neighborhood-level coverage, temporal robustness, and conformal sensitivity across 80%, 90%, and 95% nominal coverage levels.

The project is intended as a study of reliable tabular regression rather than an exhaustive model leaderboard.

## 1. Introduction

A house-price estimate is more useful when accompanied by information about how uncertain that estimate is.

This project therefore studies two related questions:

1. How accurately can residential sale prices be predicted under a leakage-safe and reproducible evaluation protocol?
2. How reliable are the associated prediction intervals, both overall and under selected subgroup and temporal stress analyses?

The project emphasizes evaluation discipline rather than maximizing a single leaderboard metric. Preprocessing, model selection, conformal calibration, final testing, and post-hoc analysis are assigned distinct roles so that information from the final test set does not influence the primary model or uncertainty procedure.

The primary prediction contract is to estimate `SalePrice` before the transaction is finalized.

## 2. Dataset and provenance

The project uses the full Ames Housing dataset.

The validated dataset contains:

- 2,930 residential property sales;
- 82 columns;
- no missing `SalePrice` values;
- 28 neighborhoods.

The raw dataset is not redistributed through this repository.

Source information, download instructions, loading semantics, and the expected SHA256 checksum are documented in:

`data/README.md`

A detailed audit is available in:

`reports/data_audit.md`

### 2.1 Missing-value semantics

The dataset contains both genuine missing values and categorical values that represent structural absence.

Literal categorical values such as `NA` are therefore preserved during loading rather than automatically interpreted as missing values.

This distinction is important for features describing garages, basements, pools, alleys, and other property components that may simply not exist.

The validated audit identified 21 columns containing genuine missing values, with 719 genuine missing cells across 661 rows.

## 3. Prediction contract and feature availability

The following columns are excluded from the primary predictor matrix:

- `SalePrice`;
- `Order`;
- `PID`;
- `Mo Sold`;
- `Yr Sold`;
- `Sale Type`;
- `Sale Condition`.

`SalePrice` is the prediction target.

`Order` and `PID` are retained as metadata rather than predictive features.

`Yr Sold` is excluded from the primary feature matrix but is retained for construction of the secondary temporal stress-test split.

Transaction-context variables are excluded because their availability is not guaranteed under the pre-sale prediction contract.

After exclusions, the primary model uses 75 predictors.

`MS SubClass` is treated as a categorical variable rather than as a continuous numeric quantity.

The complete feature decisions are documented in:

- `reports/feature_availability.md`;
- `reports/feature_schema.csv`.

## 4. Leakage-safe preprocessing

All learned preprocessing operations are fitted on training data only.

The central rule is:

```text
fit preprocessing on training data
-> transform calibration without refitting
-> transform test without refitting
```

The final preprocessing pipeline includes:

- preservation of literal structural-absence categories such as `NA`;
- train-only hierarchical imputation of `Lot Frontage`;
- explicit treatment of structural garage absence for `Garage Yr Blt`;
- train-only median imputation for remaining numeric missing values;
- an explicit `__MISSING__` category for genuine categorical missingness;
- one-hot encoding of categorical predictors;
- `handle_unknown="ignore"` behavior for previously unseen categories;
- categorical treatment of `MS SubClass`;
- optional numeric scaling for model families that require it.

### 4.1 Lot Frontage preprocessing experiment

A preliminary experiment compared several strategies for reconstructing missing `Lot Frontage` values:

- global median;
- `Lot Config` median;
- `Neighborhood` median;
- hierarchical medians;
- HistGradientBoosting-based reconstruction.

The model-based reconstruction produced the strongest reconstruction score in that isolated experiment.

However, the production pipeline deliberately uses a simpler, leakage-safe hierarchical strategy fitted on training data only:

```text
Neighborhood median
-> Lot Config median
-> global training median
```

The reconstruction experiment is treated as preprocessing research, not as evidence for changing the final point-prediction model.

## 5. Evaluation protocol

The evaluation protocol was frozen before final model testing.

### 5.1 Primary random split

The primary evaluation uses a deterministic, target-independent 60/20/20 split:

| Partition | Rows | Percentage | Role |
|---|---:|---:|---|
| Train | 1,758 | 60% | Development and model selection |
| Calibration | 586 | 20% | Conformal calibration |
| Test | 586 | 20% | Final held-out evaluation |

Partition membership is based on stable identifiers and fixed random states.

The split does not inspect `SalePrice` and does not depend on the input DataFrame row order.

Exact row membership is frozen in:

`reports/evaluation_split_manifest.csv`

The complete protocol is documented in:

`reports/evaluation_protocol.md`

### 5.2 Partition roles

The training partition is used for:

- preprocessing development;
- cross-validation;
- model-family comparison;
- hyperparameter selection;
- final point-model fitting.

The calibration partition is used only to estimate the conformal nonconformity threshold after the point model is frozen.

The test partition is reserved for the final primary evaluation.

### 5.3 Development cross-validation

Point models are compared using deterministic five-fold K-fold cross-validation on the training partition only.

Configuration:

```text
n_splits = 5
shuffle = True
random_state = 2026
```

The pre-specified primary model-selection metric is Mean Absolute Error (MAE).

RMSE is reported as a secondary metric.

## 6. Point-model selection

The model comparison intentionally includes simple, regularized linear, and nonlinear tree-based baselines rather than attempting an exhaustive state-of-the-art benchmark.

The principal development results were:

| Model | OOF MAE | OOF RMSE |
|---|---:|---:|
| Median baseline | $54,416.94 | $80,011.92 |
| Ridge | $16,063.92 | $27,776.03 |
| ElasticNet | **$16,041.52** | $27,852.31 |
| Tuned Random Forest | $16,173.00 | **$27,366.95** |

The tuned Random Forest achieved the lowest OOF RMSE.

ElasticNet achieved the lowest OOF MAE, which was the pre-specified primary selection metric, and was therefore selected as the primary point-prediction model.

The frozen ElasticNet configuration is:

```text
alpha = 0.1
l1_ratio = 0.9
target = raw SalePrice
```

No target transformation is used in the primary model.

The Ridge and ElasticNet results are very close, so the results should not be interpreted as evidence of a large performance advantage for ElasticNet. The selection follows the pre-declared metric rather than a claim of model-family dominance.

## 7. Split conformal prediction

After freezing the point model, split conformal prediction is used to construct prediction intervals.

The primary conformal procedure uses absolute residual scores:

```text
score_i = |y_i - y_hat_i|
```

For nominal coverage of 90% and a calibration set of 586 observations, the finite-sample conformal rank is:

```text
ceil((n + 1) * 0.90) = 529
```

The resulting calibration radius is:

```text
q_hat = $32,616.34
```

For a point prediction `y_hat`, the symmetric prediction interval is:

```text
[y_hat - q_hat, y_hat + q_hat]
```

The corresponding interval width is:

```text
$65,232.67
```

The 90% operating point was specified as the primary conformal setting before sensitivity analysis.

Under the standard exchangeability assumptions of split conformal prediction, the procedure targets marginal rather than subgroup-conditional coverage.

## 8. Primary held-out evaluation

The primary test set was evaluated after the point model and conformal procedure had been frozen.

The test set contains 586 observations.

### 8.1 Point-prediction performance

| Metric | Test result |
|---|---:|
| MAE | $15,957.31 |
| RMSE | $33,434.09 |

The test MAE is close to the training-only OOF estimate of $16,041.52, indicating similar typical absolute-error performance between development and final evaluation.

The test RMSE is higher than the OOF RMSE, indicating greater influence from large residuals in the held-out test partition.

### 8.2 Primary conformal performance

| Metric | Result |
|---|---:|
| Nominal coverage | 90.00% |
| Empirical coverage | **91.47%** |
| Covered observations | 536 / 586 |
| Missed observations | 50 / 586 |
| Conformal radius | $32,616.34 |
| Mean interval width | $65,232.67 |

The pre-registered 90% split-conformal interval achieved 91.47% empirical coverage on the held-out primary test set.

This is the primary uncertainty result of the project.

## 9. Subgroup reliability analysis

Neighborhood is the primary subgroup variable.

To avoid over-interpreting small samples, subgroup results use the following interpretation policy:

| Test subgroup size | Interpretation |
|---:|---|
| `n >= 50` | Primary |
| `20 <= n < 50` | Exploratory, with warning |
| `n < 20` | Descriptive only |

Among neighborhoods meeting the primary sample-size threshold:

| Neighborhood | n | Empirical coverage |
|---|---:|---:|
| CollgCr | 52 | 96.15% |
| NAmes | 81 | 97.53% |

An exploratory result of particular interest was:

| Neighborhood | n | Empirical coverage |
|---|---:|---:|
| NridgHt | 37 | 67.57% |

Because `NridgHt` contains fewer than 50 primary-test observations, this result is treated as exploratory rather than as strong evidence of systematic conditional undercoverage.

The overall conformal guarantee is marginal; subgroup-conditional coverage is not guaranteed.

## 10. Post-hoc error diagnostics

Post-hoc diagnostics were performed only after the frozen primary test evaluation.

These analyses are diagnostic and do not modify the primary model, preprocessing pipeline, conformal radius, nominal coverage, or evaluation protocol.

The main observation is that typical absolute-error performance generalized closely from OOF development estimates to the primary test set, while test RMSE increased substantially because of tail errors.

Among observations covered by the conformal interval, approximate error metrics were:

| Subset | MAE | RMSE |
|---|---:|---:|
| Covered observations | $11,264 | $14,091 |
| Conformal misses | $66,265 | $104,749 |

The largest absolute residual was approximately $594,000.

These findings show that a relatively small number of large prediction errors strongly influence RMSE and account for a disproportionate share of the most difficult cases.

Because this analysis uses the already-consumed primary test set, it is interpreted as post-hoc diagnosis rather than confirmatory model selection evidence.

## 11. Temporal stress test

A secondary temporal protocol evaluates the frozen modeling approach under a forward-looking split:

| Partition | Sale years | Rows |
|---|---|---:|
| Train | 2006-2008 | 1,941 |
| Calibration | 2009 | 648 |
| Test | 2010 | 341 |

The temporal evaluation produced:

| Metric | Result |
|---|---:|
| MAE | $16,871.76 |
| RMSE | $24,855.96 |
| Empirical coverage | 91.20% |
| Conformal radius | $36,714.44 |
| Mean interval width | $73,428.89 |

The temporal conformal rank was 585.

The procedure maintained close-to-target empirical coverage in this specific temporal stress test, although its intervals were wider than in the primary random protocol.

The temporal result should not be interpreted as a distribution-free coverage guarantee under time shift. Exchangeability is not assumed under this evaluation.

Point-error metrics from the temporal and random test sets also should not be treated as a direct model-performance ranking because the two protocols evaluate different observations.

## 12. Conformal sensitivity analysis

A secondary sensitivity analysis evaluated alternative nominal coverage levels without changing the frozen primary point model.

| Nominal coverage | Empirical coverage | Mean interval width |
|---:|---:|---:|
| 80% | 78.67% | $43,985.81 |
| **90%** | **91.47%** | **$65,232.67** |
| 95% | 95.90% | $89,303.16 |

As expected, higher nominal coverage requires wider prediction intervals.

The 90% configuration remains the project's primary operating point because it was pre-specified before this sensitivity analysis.

The 80% and 95% results are secondary analyses that illustrate the coverage-width trade-off; they are not used to retrospectively select a different primary setting.

## 13. Reproducibility and testing

The repository separates reusable source code, experiment scripts, configuration, tests, reports, and frozen result artifacts.

The project includes automated tests covering areas including:

- data loading and schema behavior;
- deterministic splitting;
- preprocessing behavior;
- leakage controls;
- model metrics and evaluation utilities;
- conformal calculations;
- integration behavior.

At the final pre-release checkpoint:

```text
python -m ruff check .
All checks passed.

python -m pytest
96 passed.
```

The final repository state is intended to allow the principal experiments and reported results to be inspected and reproduced from version-controlled code and configuration.

## 14. Interpretation

The strongest result of this project is not that one particular model dominates house-price prediction.

Instead, the project demonstrates a controlled workflow in which:

- feature availability is defined before modeling;
- preprocessing is fitted without evaluation leakage;
- model selection is restricted to the development partition;
- the primary selection metric is declared before model comparison;
- conformal calibration is separated from point-model fitting;
- the final test set is evaluated only after the primary system is frozen;
- post-hoc diagnostics are explicitly separated from confirmatory results;
- reliability is examined beyond aggregate point-prediction metrics.

The point model itself is deliberately conventional. The main emphasis is evaluation reliability and uncertainty-aware prediction.

## 15. Limitations

The study has several important limitations.

First, Ames Housing represents a historical housing market in a single geographic setting. Results therefore should not be assumed to generalize directly to current markets or other regions.

Second, the primary conformal intervals are symmetric and have constant width for all observations within a calibrated evaluation protocol. They do not adapt interval width to property-specific heteroscedasticity.

Third, split conformal prediction provides a marginal coverage target under exchangeability. It does not guarantee correct coverage for every neighborhood, price range, or other subgroup.

Fourth, several subgroup samples are small, which limits the precision of subgroup-level coverage estimates.

Fifth, the temporal experiment is a stress test under distribution shift. Its observed coverage is empirical and does not carry the standard exchangeability-based conformal guarantee.

Sixth, the model comparison is intentionally limited to a small set of baseline and regularized/tree-based model families. The project is not an exhaustive benchmark of boosting, ensembling, or other state-of-the-art tabular methods.

Finally, the primary test set has already served its intended final evaluation role. Findings from post-hoc test diagnostics should not be used to tune the frozen primary system and then be reported as new confirmatory evidence on the same test set.

## 16. Conclusion

This project develops a reproducible uncertainty-aware workflow for tabular house-price regression.

A leakage-safe preprocessing pipeline was combined with deterministic training, calibration, and test partitions. Point-model selection was performed using training-only cross-validation and a pre-specified MAE criterion, resulting in a frozen ElasticNet primary model.

On the held-out primary test set, the model achieved an MAE of $15,957.31 and an RMSE of $33,434.09. The pre-registered 90% split-conformal prediction interval achieved 91.47% empirical coverage with a mean width of $65,232.67.

Post-hoc analysis identified substantial tail errors, subgroup evaluation highlighted variation in empirical neighborhood coverage, and a temporal stress test achieved 91.20% empirical coverage with wider intervals. Sensitivity analysis further demonstrated the expected trade-off between nominal coverage and interval width.

The project therefore serves primarily as an example of disciplined machine-learning evaluation: point accuracy, uncertainty calibration, failure analysis, and distribution-shift robustness are treated as separate but connected parts of the same predictive system.
