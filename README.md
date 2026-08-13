# House Price Uncertainty

A reproducible machine-learning project for **house-price prediction, uncertainty estimation, subgroup diagnostics, and leakage-safe evaluation**.

> **Status:** Active development — data validation, leakage auditing, and the evaluation protocol are complete; leakage-safe preprocessing is next.

## Why this project exists

A point estimate alone can be misleading. A useful housing model should communicate both:

1. the predicted sale price, and
2. how uncertain that prediction is.

This project therefore evaluates not only point-prediction error, but also the empirical coverage and width of prediction intervals across different price ranges, neighborhoods, and evaluation settings.

The primary prediction contract estimates the expected residential market value of a property **before its transaction is finalized**.

## Research questions

1. How much do regularized linear models and tree-based models improve over a simple median baseline?
2. Does a log-transformed target improve predictive performance or residual behavior?
3. Do nominal prediction intervals achieve their intended empirical coverage on held-out data?
4. Is interval coverage consistent across price bands and neighborhoods?
5. How do point-prediction and interval performance change under a forward-looking temporal distribution shift?
6. Which property characteristics contribute most strongly to predictions, and where does the model fail?

## Planned models

* Median-prediction baseline
* Ridge / Elastic Net
* Gradient boosting or CatBoost
* Split-conformal prediction wrapper for calibrated prediction intervals

## Primary metrics

### Point prediction

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Uncertainty

* Empirical interval coverage
* Mean prediction-interval width
* Coverage and interval width by subgroup

## Dataset

The project uses the full Ames Housing dataset:

* 2,930 residential property sales
* 82 columns
* prediction target: `SalePrice`
* 28 neighborhoods
* no missing target values

The raw dataset is not committed to the repository.

Dataset provenance, download instructions, loading semantics, checksum verification, citation information, and redistribution considerations are documented in [`data/README.md`](data/README.md).

A detailed audit is available in [`reports/data_audit.md`](reports/data_audit.md).

## Missing-data semantics

Ames Housing contains both genuine missing values and categorical values such as `NA` that can represent **structural absence**.

For example, `NA` may indicate that a property has no garage, basement, pool, or related feature rather than that the value is unknown.

The dataset loader therefore preserves literal categorical `NA` values instead of automatically converting them to missing values.

The project follows the principle:

> **Structural absence is not the same as unknown missingness.**

Missing-value treatment is being developed as part of the leakage-safe preprocessing pipeline.

A dedicated `Lot Frontage` reconstruction benchmark has already compared global, group-aware, hierarchical, and model-based imputation strategies. These results are treated as preprocessing research rather than as a final production decision.

## Leakage-safe prediction contract

The primary model estimates expected residential market value before the transaction is finalized.

The following columns are excluded from the primary predictor matrix:

* `Order`
* `PID`
* `Mo Sold`
* `Yr Sold`
* `Sale Type`
* `Sale Condition`
* `SalePrice`

`SalePrice` is the prediction target.

`Order` and `PID` are retained as metadata rather than predictive features.

Transaction-context variables are excluded from the primary pre-sale model because their availability is not guaranteed at prediction time. They are not assumed to be inherently invalid and may be evaluated later through explicitly defined sensitivity analyses.

Feature-availability and semantic-role decisions are documented in:

* [`reports/feature_availability.md`](reports/feature_availability.md)
* [`reports/feature_schema.csv`](reports/feature_schema.csv)

## Evaluation protocol

The evaluation design is frozen before final model fitting.

### Primary protocol

A deterministic, target-independent random split is used:

| Partition   |  Rows | Percentage |
| ----------- | ----: | ---------: |
| Train       | 1,758 |        60% |
| Calibration |   586 |        20% |
| Test        |   586 |        20% |

Partition membership is generated from stable `Order` identifiers using fixed random states `42` and `43`.

The split:

* does not require or inspect `SalePrice`;
* is independent of input DataFrame row order;
* is reproducible from the documented algorithm and fixed seeds.

Model development, preprocessing selection, cross-validation, model-family comparison, and hyperparameter tuning remain inside the training partition.

The calibration partition is reserved for conformal calibration.

The test partition is reserved for final point-prediction, interval, and subgroup evaluation.

### Temporal stress test

A forward-looking secondary protocol evaluates performance under temporal distribution shift:

| Partition   | Sale years |  Rows |
| ----------- | ---------- | ----: |
| Train       | 2006–2008  | 1,941 |
| Calibration | 2009       |   648 |
| Test        | 2010       |   341 |

`Yr Sold` is used to construct this protocol but is not included in the primary predictor matrix.

The temporal protocol is treated as a stress test. Empirical interval coverage will be reported, but a distribution-free conformal guarantee is not assumed under temporal distribution shift.

The complete evaluation policy is documented in [`reports/evaluation_protocol.md`](reports/evaluation_protocol.md).

Exact row assignments are frozen in [`reports/evaluation_split_manifest.csv`](reports/evaluation_split_manifest.csv).

## Subgroup evaluation policy

Neighborhood-level interval results will be interpreted according to subgroup size:

| Test subgroup size | Interpretation                                         |
| -----------------: | ------------------------------------------------------ |
|          `n >= 50` | Primary subgroup interpretation                        |
|     `20 <= n < 50` | Exploratory interpretation with a small-sample warning |
|           `n < 20` | Raw count and coverage only; no strong conclusion      |

Small subgroups are retained rather than removed from evaluation.

Final subgroup reports will include sample size, empirical coverage, and interval width where appropriate.

## Repository structure

```text
house-price-uncertainty/
├── configs/                 # Model and experiment configuration
├── data/                    # Dataset provenance and download instructions
├── experiments/             # Reproducible experiments and generated results
├── notebooks/               # Exploration and presentation only
├── reports/                 # Audits, schemas, protocol, and technical report
├── src/house_price_uncertainty/
│   ├── data.py              # Dataset loading and integrity validation
│   ├── feature_schema.py    # Leakage-safe feature roles and preparation
│   ├── metrics.py           # Point and interval metrics
│   ├── splitting.py         # Reproducible evaluation protocols
│   └── validation.py        # Shared validation helpers
├── tests/                   # Automated tests
└── .github/workflows/       # Continuous integration
```

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

## Reproducing current project artifacts

Feature-schema and evaluation-manifest artifacts can be regenerated with:

```bash
python experiments/build_feature_schema.py
python experiments/build_evaluation_manifest.py
```

The `Lot Frontage` reconstruction benchmark is implemented in:

```text
experiments/lot_frontage_imputation_benchmark.py
```

Machine-readable experiment outputs are stored under:

```text
experiments/results/
```

## Planned output format

The final project will produce outputs similar to:

```text
Predicted price: $212,000
90% prediction interval: $188,000–$239,000
```

The values above illustrate the intended output format and are **not reported model results**.

## Planned command-line interface

A later release may expose training and evaluation commands similar to:

```bash
python -m house_price_uncertainty.train --config configs/baseline.yaml
python -m house_price_uncertainty.evaluate --config configs/baseline.yaml
```

These commands are not implemented yet.

## Project history

This project began as an educational machine-learning exercise and is being redesigned as a reproducible, research-oriented ML project.

The redesign includes both completed work and planned extensions in:

* dataset provenance and integrity validation;
* explicit missing-data semantics;
* leakage and prediction-time feature auditing;
* reproducible train/calibration/test protocols;
* temporal stress testing;
* automated testing and continuous integration;
* uncertainty estimation and subgroup reliability analysis.

## What did not work

Negative and inconclusive results are documented rather than omitted.

Current preprocessing research has shown that simple global median imputation is a weak reconstruction strategy for `Lot Frontage` compared with neighborhood-aware, hierarchical, and model-based alternatives.

However, reconstruction error alone does not determine the final production strategy. Final preprocessing decisions will also consider downstream `SalePrice` prediction, uncertainty performance, stability, and implementation complexity.

## Current roadmap

* [x] Define the research question and publication standard
* [x] Create the repository scaffold
* [x] Add tested point and interval metrics
* [x] Validate the dataset, licensing, and evaluation split
* [ ] Write a leakage-safe preprocessing pipeline
* [ ] Establish median and regularized-linear baselines
* [ ] Train a tree-based model
* [ ] Add conformal prediction
* [ ] Run subgroup coverage analysis
* [ ] Complete the technical report
* [ ] Create release `v1.0.0`

## License

Project code is released under the MIT License.

Dataset provenance, citation, usage, and redistribution considerations are documented separately in [`data/README.md`](data/README.md).
