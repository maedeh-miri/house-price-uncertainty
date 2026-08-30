# House Price Uncertainty

A reproducible machine-learning project for **house-price prediction, uncertainty estimation, subgroup diagnostics, and leakage-safe evaluation**.

> **Status:** Released — `v1.0.0` is the first stable release. The primary evaluation, post-hoc diagnostics, temporal stress test, conformal sensitivity analysis, technical report, and release documentation are complete.

## Why this project exists

A point estimate alone can be misleading. A useful housing model should communicate both:

1. the predicted sale price; and
2. how uncertain that prediction is.

This project therefore evaluates not only point-prediction error, but also the empirical coverage and width of prediction intervals, subgroup reliability, and performance under temporal distribution shift.

The primary prediction contract estimates the expected residential market value of a property **before its transaction is finalized**.

For the complete methodology and final findings, see [`reports/technical_report.md`](reports/technical_report.md).

## Research questions

1. How much do regularized linear models and tree-based models improve over a simple median baseline?
2. Can a split-conformal prediction system achieve approximately its pre-specified 90% marginal coverage on held-out data?
3. Is empirical interval coverage consistent across neighborhoods?
4. Why do some held-out observations produce much larger prediction errors than others?
5. How do point-prediction and interval performance change under a forward-looking temporal distribution shift?

## Modeling approach

The project evaluates a deliberately small set of point-prediction baselines and candidate models:

- median-prediction baseline;
- Ridge regression;
- ElasticNet;
- tuned Random Forest.

Model selection is performed using cross-validation inside the training partition only.

The selected primary point model is:

```text
ElasticNet
alpha = 0.1
l1_ratio = 0.9
target = raw SalePrice
target transformation = none
```

ElasticNet was selected because MAE was defined as the primary model-selection metric before comparing the candidate models.

A split-conformal wrapper is then calibrated around the frozen ElasticNet point predictor to produce symmetric prediction intervals.

## Primary evaluation results

The final primary evaluation was performed once on the frozen 586-row test partition after preprocessing, model selection, conformal specification, and calibration had been committed.

### Point prediction

| Metric | Test result |
|---|---:|
| MAE | **$15,957.31** |
| RMSE | **$33,434.09** |

### 90% split-conformal prediction intervals

| Metric | Test result |
|---|---:|
| Nominal coverage | 90.00% |
| Empirical coverage | **91.47%** |
| Covered observations | **536 / 586** |
| Conformal radius | **$32,616.34** |
| Mean interval width | **$65,232.67** |

The conformal calibration used:

```text
Calibration rows: 586
Finite-sample quantile rank: 529
q_hat: 32616.33610273435
Interval type: symmetric
Nonconformity score: absolute residual
```

The primary test set is now considered **consumed and frozen**.

Its results may be analyzed diagnostically, but the primary model, hyperparameters, preprocessing rules, conformal radius, nominal coverage target, or evaluation split will not be changed in response to test-set performance.

## Development results

Model comparison was performed with deterministic 5-fold out-of-fold evaluation using the training partition only.

| Model | OOF MAE | OOF RMSE |
|---|---:|---:|
| Median baseline | $54,416.94 | $80,011.92 |
| Ridge | $16,063.92 | $27,776.03 |
| **ElasticNet** | **$16,041.52** | $27,852.31 |
| Tuned Random Forest | $16,173.00 | **$27,366.95** |

ElasticNet and Ridge were nearly tied on MAE.

Random Forest achieved the lowest OOF RMSE, but MAE had been declared the primary selection metric before model comparison, so ElasticNet remained the selected point model.

These OOF results are development estimates and are not presented as final test performance.

## Subgroup diagnostics

Neighborhood-level interval coverage is interpreted according to a sample-size policy that was frozen before final evaluation:

| Test subgroup size | Interpretation |
|---:|---|
| `n >= 50` | Primary subgroup interpretation |
| `20 <= n < 50` | Exploratory interpretation with a small-sample warning |
| `n < 20` | Descriptive result only |

Two neighborhoods reached the threshold for primary interpretation in the final test partition:

| Neighborhood | n | Covered | Empirical coverage |
|---|---:|---:|---:|
| CollgCr | 52 | 50 | 96.15% |
| NAmes | 81 | 79 | 97.53% |

Several smaller neighborhoods showed more variable empirical coverage. For example, `NridgHt` had 67.57% empirical coverage with `n = 37`, so it is treated as exploratory rather than as a primary subgroup conclusion.

These subgroup results are diagnostic only. Standard split conformal targets **marginal coverage** and does not provide a guarantee of equal conditional coverage within every neighborhood.

## Dataset

The project uses the full Ames Housing dataset:

- 2,930 residential property sales;
- 82 columns;
- prediction target: `SalePrice`;
- 28 neighborhoods;
- no missing target values.

The raw dataset is not committed to the repository.

Dataset provenance, download instructions, loading semantics, checksum verification, citation information, and redistribution considerations are documented in [`data/README.md`](data/README.md).

A detailed audit is available in [`reports/data_audit.md`](reports/data_audit.md).

## Missing-data semantics

Ames Housing contains both genuine missing values and categorical values such as `NA` that can represent **structural absence**.

For example, `NA` may indicate that a property has no garage, basement, pool, or related feature rather than that the value is unknown.

The dataset loader therefore preserves literal categorical `NA` values instead of automatically converting them to missing values.

The project follows the principle:

> **Structural absence is not the same as unknown missingness.**

The leakage-safe preprocessing pipeline distinguishes structural absence from genuinely unknown missing values and learns all data-dependent transformations using the training partition only.

A dedicated `Lot Frontage` reconstruction benchmark compared global, group-aware, hierarchical, and model-based imputation strategies.

Although model-based reconstruction achieved lower reconstruction error, the production preprocessing baseline uses a simpler train-only hierarchical strategy based on:

```text
Neighborhood
→ Lot Config
→ global training median
```

This keeps the production pipeline interpretable while avoiding leakage from calibration or test data.

## Leakage-safe prediction contract

The primary model estimates expected residential market value before the transaction is finalized.

The following columns are excluded from the primary predictor matrix:

- `Order`
- `PID`
- `Mo Sold`
- `Yr Sold`
- `Sale Type`
- `Sale Condition`
- `SalePrice`

`SalePrice` is the prediction target.

`Order` and `PID` are retained as metadata rather than predictive features.

Transaction-context variables are excluded from the primary pre-sale model because their availability is not guaranteed at prediction time. They are not assumed to be inherently invalid and could be studied later through explicitly defined sensitivity analyses.

Feature-availability and semantic-role decisions are documented in:

- [`reports/feature_availability.md`](reports/feature_availability.md)
- [`reports/feature_schema.csv`](reports/feature_schema.csv)

## Leakage-safe preprocessing

All learned preprocessing operations are fitted on training data only.

Calibration and test partitions are transformed without refitting preprocessing statistics.

The pipeline includes:

- structural-absence handling;
- train-only missing-value imputation;
- hierarchical `Lot Frontage` imputation;
- garage-year handling;
- categorical missing-value representation;
- one-hot encoding with unseen-category support;
- model-dependent numeric scaling.

Unseen categorical levels are handled without changing the accepted evaluation split.

The preprocessing implementation is designed to support both linear and tree-based estimators without allowing calibration or test information to influence fitted transformations.

## Evaluation protocol

The evaluation design was frozen before final model evaluation.

### Primary protocol

A deterministic, target-independent random split is used:

| Partition | Rows | Percentage |
|---|---:|---:|
| Train | 1,758 | 60% |
| Calibration | 586 | 20% |
| Test | 586 | 20% |

Partition membership is generated from stable `Order` identifiers using fixed random states `42` and `43`.

The split:

- does not require or inspect `SalePrice`;
- is independent of input DataFrame row order;
- is reproducible from the documented algorithm and fixed seeds.

Partition roles are strictly separated.

### Train

Used for:

- fitting preprocessing transformations;
- cross-validation;
- model-family comparison;
- hyperparameter selection;
- final point-model fitting.

### Calibration

Reserved for:

- computing absolute residual nonconformity scores;
- estimating the split-conformal quantile.

It is not used for model selection or hyperparameter tuning.

### Test

Reserved for:

- final MAE and RMSE;
- final empirical interval coverage;
- interval width;
- subgroup diagnostics.

The primary test partition has now been evaluated once and is considered consumed.

### Temporal stress test

A forward-looking secondary protocol was evaluated:

| Partition | Sale years | Rows |
|---|---|---:|
| Train | 2006–2008 | 1,941 |
| Calibration | 2009 | 648 |
| Test | 2010 | 341 |

`Yr Sold` is used to construct this protocol but is not included in the primary predictor matrix.

The temporal protocol is treated as a distribution-shift stress test.

The completed temporal evaluation produced:

| Metric | Result |
|---|---:|
| MAE | **$16,871.76** |
| RMSE | **$24,855.96** |
| Empirical coverage | **91.20%** |
| Mean interval width | **$73,428.89** |

The result indicates that the conformal uncertainty procedure achieved close-to-target **empirical coverage** under this forward-looking temporal stress test.

The temporal protocol is a secondary robustness analysis. The usual exchangeability-based split-conformal guarantee is not assumed under temporal distribution shift.

The complete evaluation policy is documented in [`reports/evaluation_protocol.md`](reports/evaluation_protocol.md).

Exact row assignments are frozen in [`reports/evaluation_split_manifest.csv`](reports/evaluation_split_manifest.csv).

## Conformal sensitivity analysis

A secondary sensitivity analysis evaluated different nominal coverage levels while keeping the frozen point model and conformal procedure unchanged.

| Nominal coverage | Empirical coverage | Mean interval width |
|---:|---:|---:|
| 80% | 78.67% | $43,985.81 |
| **90%** | **91.47%** | **$65,232.67** |
| 95% | 95.90% | $89,303.16 |

The 90% conformal configuration remains the primary uncertainty setting because it was pre-specified before the sensitivity analysis.

The 80% and 95% configurations illustrate the expected trade-off between empirical coverage and interval width. The 95% configuration achieves higher empirical coverage at the cost of substantially wider prediction intervals.

## Conformal prediction protocol

The primary uncertainty method is frozen as:

```text
Method: split conformal
Score: absolute residual
Nominal coverage: 90%
Interval type: symmetric
Lower-bound clipping: disabled
```

For the primary calibration partition:

```text
n_calibration = 586
quantile rank = ceil((586 + 1) * 0.90)
              = 529
```

The resulting radius is:

```text
q_hat = $32,616.34
```

For a point prediction `ŷ`, the reported interval is:

```text
[ŷ - q_hat, ŷ + q_hat]
```

Calibration-set coverage itself is not treated as an independent performance estimate.

Final performance is evaluated on the held-out test partition.

## Repository structure

```text
house-price-uncertainty/
├── configs/                 # Model and conformal configuration
├── data/                    # Dataset provenance and download instructions
├── experiments/             # Reproducible experiments and generated results
├── notebooks/               # Exploration and presentation only
├── reports/                 # Audits, schemas, protocol, and technical report
├── src/house_price_uncertainty/
│   ├── conformal.py         # Split-conformal calibration and intervals
│   ├── data.py              # Dataset loading and integrity validation
│   ├── feature_schema.py    # Leakage-safe feature roles and preparation
│   ├── metrics.py           # Point and interval metrics
│   ├── models.py            # Model-building utilities
│   ├── preprocessing.py     # Leakage-safe preprocessing pipelines
│   ├── splitting.py         # Reproducible evaluation protocols
│   └── validation.py        # Shared validation helpers
├── tests/                   # Automated tests
└── .github/workflows/       # Continuous integration
```

## Reproducibility and testing

The repository contains **96 passing automated tests** covering the core data, feature-schema, splitting, preprocessing, modeling, metric, and conformal utilities.

Run the full validation suite with:

```bash
python -m ruff check src tests experiments
python -m pytest
```

The primary experiment artifacts are machine-readable and committed so that major modeling and evaluation decisions can be audited from Git history.

## Quick start

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check src tests experiments
python -m pytest
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ruff check src tests experiments
python -m pytest
```

## Reproducing project artifacts

Feature-schema and evaluation-manifest artifacts can be regenerated with:

```bash
python experiments/build_feature_schema.py
python experiments/build_evaluation_manifest.py
```

The `Lot Frontage` reconstruction benchmark is implemented in:

```text
experiments/lot_frontage_imputation_benchmark.py
```

Primary conformal calibration is implemented in:

```text
experiments/conformal_calibration.py
```

The frozen final primary evaluation is implemented in:

```text
experiments/final_primary_evaluation.py
```

Post-hoc diagnostics are implemented in:

```text
experiments/posthoc_diagnostics.py
```

The temporal stress test is implemented in:

```text
experiments/temporal_stress_test.py
```

Conformal coverage sensitivity analysis is implemented in:

```text
experiments/conformal_sensitivity.py
```

Machine-readable outputs are stored under:

```text
experiments/results/
```

Important committed artifacts include:

```text
experiments/results/conformal_calibration_summary.json
experiments/results/final_primary_test_summary.json
experiments/results/posthoc_diagnostics_summary.json
experiments/results/temporal_stress_summary.json
experiments/results/conformal_sensitivity_summary.json
```

## Example interpretation

The selected model produces a point prediction together with a 90% symmetric conformal interval.

For illustration, a prediction of:

```text
Predicted price: $200,000
```

would use the frozen primary conformal radius to produce approximately:

```text
90% prediction interval:
$167,384 – $232,616
```

The interval width is constant for the current symmetric absolute-residual conformal specification.

## Project history

This project began as an educational machine-learning exercise and was redesigned as a reproducible, research-oriented ML project.

The redesign includes:

- dataset provenance and integrity validation;
- explicit missing-data semantics;
- leakage and prediction-time feature auditing;
- reproducible train/calibration/test protocols;
- leakage-safe preprocessing;
- controlled point-model comparison;
- automated testing and continuous integration;
- split-conformal uncertainty estimation;
- frozen final test evaluation;
- subgroup reliability diagnostics;
- post-hoc residual and tail-error diagnostics;
- temporal stress testing as a completed secondary robustness analysis;
- conformal coverage sensitivity analysis across 80%, 90%, and 95% nominal levels;
- a finalized technical report and stable `v1.0.0` release.

## What did not work or did not win

Negative and inconclusive findings are documented rather than omitted.

### `Lot Frontage` reconstruction

Simple global median imputation produced substantially worse reconstruction error than neighborhood-aware, hierarchical, and model-based approaches.

A HistGradientBoosting reconstruction model performed best in the dedicated reconstruction benchmark, but was not automatically adopted as the production solution because downstream simplicity, stability, leakage control, and interpretability also matter.

### Ridge versus ElasticNet

Ridge and ElasticNet achieved nearly identical OOF MAE.

ElasticNet won by only a small margin, so the project does not claim a substantial performance advantage over Ridge.

### Random Forest

The tuned Random Forest achieved the lowest OOF RMSE but did not win on the pre-specified primary selection metric, MAE.

The evaluation rule was not changed after observing this result.

## Limitations

Current limitations include:

- the conformal intervals are symmetric and have constant width;
- split conformal targets marginal rather than subgroup-conditional coverage;
- several neighborhoods have small held-out sample sizes;
- final-test RMSE is noticeably larger than development OOF RMSE;
- tail-error analysis showed that a small number of large residuals strongly influence RMSE;
- temporal evaluation was performed as a secondary stress test rather than as the primary selection protocol;
- the current primary model does not use a transformed target;
- the project focuses on predictive reliability rather than causal interpretation;
- Ames Housing represents a historical housing market in a single geographic setting;
- the model comparison is deliberately limited and is not an exhaustive benchmark of state-of-the-art tabular methods.

## Project roadmap

- [x] Define the research question and publication standard
- [x] Create the repository scaffold
- [x] Validate dataset provenance and integrity
- [x] Complete data and missingness audit
- [x] Complete feature-availability and leakage audit
- [x] Freeze primary and temporal evaluation protocols
- [x] Build the leakage-safe preprocessing pipeline
- [x] Establish the median baseline
- [x] Evaluate regularized linear models
- [x] Evaluate a tuned tree-based model
- [x] Select the primary point-prediction model
- [x] Implement and test split-conformal utilities
- [x] Freeze the conformal calibration protocol
- [x] Calibrate the 90% primary conformal interval
- [x] Freeze final interval evaluation metrics
- [x] Run and record the final primary test evaluation
- [x] Record Neighborhood-level coverage diagnostics
- [x] Complete post-hoc residual and tail-error diagnostics
- [x] Run the temporal stress-test protocol
- [x] Run secondary 80% / 90% / 95% coverage sensitivity analysis
- [x] Complete the technical report
- [x] Finalize release documentation and repository cleanup
- [x] Create release `v1.0.0`

## License

Project code is released under the MIT License.

Dataset provenance, citation, usage, and redistribution considerations are documented separately in [`data/README.md`](data/README.md).
