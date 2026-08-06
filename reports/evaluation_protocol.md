# Evaluation Protocol

## 1. Purpose

This document defines the frozen evaluation protocol for the Ames
Housing price-prediction project.

The protocol separates model development, uncertainty calibration,
and final evaluation so that reported point-prediction and interval
results are not influenced by test-set feedback.

Two evaluation settings are used:

1. a target-independent random split as the primary protocol;
2. a forward-looking temporal split as a distribution-shift stress
   test.

Exact row assignments are stored in the evaluation split manifest.

## 2. Prediction contract

The primary model estimates the expected residential market value of
a property before its transaction is finalized.

The primary predictor matrix therefore uses property information that
is expected to be available before sale completion.

The following columns are excluded from the primary predictor matrix:

- `Order`
- `PID`
- `Mo Sold`
- `Yr Sold`
- `Sale Type`
- `Sale Condition`
- `SalePrice`

`SalePrice` is the prediction target.

`Order` and `PID` are retained only as metadata and stable row
identifiers.

The transaction-context variables are excluded from the primary
pre-sale model because their availability is not guaranteed at
prediction time. They are not assumed to be inherently invalid.
Separate sensitivity analyses may evaluate market-aware and
transaction-aware variants.

`Yr Sold` is retained outside the predictor matrix so that it can be
used to construct the temporal stress-test protocol.

## 3. Partition roles

Every protocol contains three mutually exclusive partitions.

### Training partition

The training partition may be used for:

- fitting preprocessing transformations;
- feature engineering development;
- cross-validation;
- model-family comparison;
- hyperparameter selection;
- final point-model fitting.

All learned preprocessing parameters must be fitted using training
data only.

Examples include:

- imputation statistics;
- category mappings;
- feature scaling parameters;
- rare-category rules;
- transformations;
- model coefficients and trees.

Cross-validation and model selection must remain entirely inside the
training partition.

### Calibration partition

The calibration partition is reserved for conformal calibration.

It may be used to estimate nonconformity-score quantiles after the
point-prediction model and its preprocessing pipeline have been
fitted.

It must not be used for:

- selecting preprocessing rules;
- selecting model families;
- tuning hyperparameters;
- choosing among candidate models based on predictive performance.

Using calibration outcomes for these decisions would weaken the
independence required by split-conformal evaluation.

### Test partition

The test partition is reserved for final evaluation.

It will be used to report:

- point-prediction metrics;
- interval coverage;
- interval width;
- subgroup coverage;
- temporal stress-test performance.

After this protocol is accepted, test targets must not be used to
choose preprocessing rules, model families, hyperparameters,
conformal methods, or random seeds.

## 4. Primary protocol: target-independent random split

The primary protocol uses a deterministic random split:

| Partition | Rows | Percentage |
|---|---:|---:|
| Train | 1,758 | 60% |
| Calibration | 586 | 20% |
| Test | 586 | 20% |

The split is generated in two stages:

1. Stable `Order` identifiers are sorted.
2. A 60/40 split is generated using random state `42`.
3. The 40% remainder is divided equally using random state `43`.

This produces the final 60/20/20 allocation.

### Target independence

The random split function does not require or inspect `SalePrice`.

Changing, permuting, or removing target values therefore does not
change partition membership.

This prevents target distribution information from determining the
primary row assignment.

### Row-order independence

Stable `Order` identifiers are sorted before random sampling.

Therefore, reordering the input DataFrame does not change which rows
belong to training, calibration, or test.

### Reproducibility

The random states and splitting procedure are fixed before model
evaluation.

They must not be changed in response to:

- model accuracy;
- conformal coverage;
- interval width;
- subgroup results;
- unusual category placement;
- visually preferable target distributions.

Selecting a different seed after inspecting results would constitute
split shopping and could produce an optimistically selected
evaluation.

## 5. Secondary protocol: temporal stress test

The forward-looking temporal protocol is:

| Partition | Sale years | Rows | Percentage |
|---|---|---:|---:|
| Train | 2006–2008 | 1,941 | 66.25% |
| Calibration | 2009 | 648 | 22.12% |
| Test | 2010 | 341 | 11.64% |

This protocol asks whether a model trained on earlier years remains
accurate and well calibrated in a later year.

`Yr Sold` determines partition membership for this protocol but is
not provided to the primary model as a predictor.

The temporal protocol represents a distribution-shift stress test.
The 2009 calibration observations and 2010 test observations may not
follow the same distribution.

For this reason, conformal coverage in the temporal protocol will be
reported as empirical coverage. A distribution-free coverage
guarantee is not assumed under temporal distribution shift.

## 6. Random-split diagnostics

The accepted target-independent random split produced the following
target quantiles:

| Quantile | Full data | Train | Calibration | Test |
|---:|---:|---:|---:|---:|
| 1% | 61,756 | 64,785 | 60,000 | 58,369 |
| 5% | 87,500 | 87,000 | 87,138 | 89,125 |
| 50% | 160,000 | 160,100 | 164,200 | 159,000 |
| 95% | 335,000 | 325,680 | 340,750 | 337,375 |
| 99% | 456,666 | 453,262 | 475,000 | 445,390 |

Upper-tail observations occur in every partition:

| Target region | Train | Calibration | Test |
|---|---:|---:|---:|
| Upper 5% | 81 | 34 | 33 |
| Upper 1% | 17 | 7 | 6 |

The observed median prices were:

| Partition | Median SalePrice |
|---|---:|
| Train | 160,100 |
| Calibration | 164,200 |
| Test | 159,000 |

These diagnostics were used to document the accepted split, not to
select among alternative random seeds.

## 7. Temporal-split diagnostics

The temporal protocol produced the following target summaries:

| Partition | Rows | Median SalePrice | Mean SalePrice |
|---|---:|---:|---:|
| Train | 1,941 | 161,900 | 182,033 |
| Calibration | 648 | 160,850 | 181,405 |
| Test | 341 | 155,000 | 172,598 |

The lower median and mean prices in the 2010 test partition indicate
a possible temporal distribution shift.

This difference is part of the intended stress test and must not be
removed by changing temporal boundaries after model evaluation.

## 8. Unseen categorical levels

Some categorical levels occur in calibration or test but not in the
random-protocol training partition.

Observed calibration-only levels include:

- `MS Zoning`: `I (all)`
- `Condition 2`: `RRNn`
- `Exterior 1st`: `PreCast`
- `Exterior 2nd`: `Other`, `PreCast`
- `Kitchen Qual`: `Po`
- `Pool QC`: `Fa`
- `Misc Feature`: `TenC`

Observed test-only levels include:

- `MS SubClass`: `150`
- `MS Zoning`: `I (all)`
- `Condition 2`: `RRAe`
- `Roof Matl`: `ClyTile`, `Roll`
- `Pool QC`: `Fa`

These observations are not treated as reasons to choose a different
random seed.

The preprocessing pipeline must explicitly support categories that
were not observed during training. Transforming calibration or test
data must not fail because of an unseen level.

Category handling rules must be learned or configured without using
calibration or test target values.

## 9. Subgroup coverage policy

Neighborhood-level interval coverage will be interpreted according
to test subgroup size:

| Test subgroup size | Reporting policy |
|---:|---|
| `n >= 50` | Primary subgroup interpretation |
| `20 <= n < 50` | Exploratory interpretation with a small-sample warning |
| `n < 20` | Raw sample count and coverage only; no strong conclusion |

Small groups are not removed from evaluation.

This policy limits the strength of conclusions drawn from unstable
coverage proportions. For example, when `n = 20`, one observation
changes the estimated coverage by five percentage points.

Subgroup reports should include both:

- the number of observations;
- the observed coverage estimate.

Interval width should also be reported where useful because nominal
coverage alone does not show whether intervals are practically
informative.

## 10. Frozen split manifest

Exact row membership is stored in:

`reports/evaluation_split_manifest.csv`

The manifest contains:

- `Order`;
- `Yr Sold`;
- random-protocol partition membership;
- temporal-protocol partition membership.

Each source row must appear exactly once in each protocol.

The manifest provides a durable record of partition membership in
addition to the documented splitting algorithm and random states.

Once committed, the manifest is the authoritative split assignment
for version 1 of the project.

Changing random states, temporal boundaries, partition proportions,
or individual row assignments requires an explicit protocol revision.
Such changes must not be made in response to model performance.

## 11. Automated validation

Automated tests verify that:

- random partitions have the expected sizes;
- partitions are mutually disjoint;
- partitions cover every source row;
- fixed random states reproduce identical membership;
- membership is independent of DataFrame row order;
- the random split does not require the prediction target;
- partition rows use stable ordering and reset indices;
- temporal partitions respect chronological boundaries;
- missing or duplicate row identifiers fail explicitly;
- missing or nonnumeric temporal values fail explicitly;
- uncovered temporal years fail explicitly;
- invalid temporal boundaries fail explicitly.

These tests protect the evaluation protocol against accidental
changes during later development.

## 12. Protocol freeze

This protocol is frozen before fitting the final SalePrice models.

The intended workflow is:

1. develop and compare models using training data only;
2. fit the selected point-prediction pipeline on the complete
   training partition;
3. estimate conformal scores using the calibration partition;
4. evaluate point and interval performance once on the test
   partition;
5. repeat the same model specification under the temporal stress-test
   protocol;
6. report limitations, subgroup sample sizes, empirical coverage, and
   interval width without changing the accepted splits.