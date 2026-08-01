# House Price Uncertainty

A reproducible machine-learning project for **house-price prediction, uncertainty estimation, subgroup diagnostics, and leakage-safe evaluation**.

> **Status:** Active development — project scaffold complete; dataset validation is next.

## Why this project exists

A point estimate alone can be misleading. A useful housing model should communicate both:

1. the predicted sale price, and
2. how uncertain that prediction is.

The project therefore evaluates not only predictive error, but also the empirical coverage and width of prediction intervals across different price ranges and neighborhoods.

## Research questions

1. How much do regularized linear models and gradient-boosted trees improve over a simple baseline?
2. Does training on a log-transformed target improve residual behavior?
3. Do nominal 90% prediction intervals actually cover about 90% of held-out prices?
4. Is interval coverage consistent across price bands and neighborhoods?
5. Which property features drive predictions, and where does the model fail?

## Planned models

- Median-prediction baseline
- Ridge / Elastic Net
- Gradient boosting or CatBoost
- Conformal prediction wrapper for 90% intervals

## Primary metrics

### Point prediction

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Uncertainty

- Empirical interval coverage
- Mean prediction-interval width
- Coverage and width by subgroup

## Evaluation protocol

- A final test set is isolated before model selection.
- Preprocessing is fitted inside cross-validation pipelines.
- Hyperparameters are selected only on training folds.
- Conformal calibration uses data not used to fit the underlying model.
- Important results are repeated across multiple seeds where applicable.

## Repository structure

```text
house-price-uncertainty/
├── configs/                 # Experiment configuration
├── data/                    # Data instructions; raw data is not committed
├── experiments/             # Metrics and experiment notes
├── notebooks/               # EDA and result presentation only
├── reports/                 # Technical report and figures
├── src/house_price_uncertainty/
│   ├── metrics.py           # Point and interval metrics
│   └── validation.py        # Input validation helpers
├── tests/                   # Automated tests
└── .github/workflows/       # Continuous integration
```

## Quick start

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## Planned output format

```text
Predicted price: $212,000
90% prediction interval: $188,000–$239,000
```

## Planned command-line interface

The final release will support commands similar to:

```bash
python -m house_price_uncertainty.train --config configs/baseline.yaml
python -m house_price_uncertainty.evaluate --config configs/baseline.yaml
```

## Project history

This project extends an earlier educational exercise with a redesigned evaluation protocol, uncertainty estimation, automated testing, and reproducible experiment workflows.

## What did not work

Negative and inconclusive results will be documented here as experiments are completed.

## Current roadmap

- [x] Define the research question and publication standard
- [x] Create the repository scaffold
- [x] Add tested point and interval metrics
- [ ] Validate the dataset, licensing, and evaluation split
- [ ] Write a leakage-safe preprocessing pipeline
- [ ] Establish median and regularized-linear baselines
- [ ] Train a tree-based model
- [ ] Add conformal prediction
- [ ] Run subgroup coverage analysis
- [ ] Write the technical report
- [ ] Create release `v1.0.0`

## License

Code is released under the MIT License. Dataset licensing and redistribution terms will be documented separately in `data/README.md`.
