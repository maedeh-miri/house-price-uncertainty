# Experiment 009 — Temporal Stress Test Results

## Objective

This experiment evaluates whether the frozen primary prediction and
uncertainty pipeline remains reliable under temporal distribution shift.

The experiment does not modify the model, preprocessing, conformal
method, or hyperparameters.

## Frozen System

Point model:

- Model: ElasticNet
- alpha: 0.1
- l1_ratio: 0.9
- Target: SalePrice
- Target transformation: none

Conformal prediction:

- Method: split conformal prediction
- Score: absolute residual
- Nominal coverage: 90%
- Interval type: symmetric

## Temporal Protocol

The data is split chronologically:

| Partition | Years |
|---|---|
| Train | 2006–2008 |
| Calibration | 2009 |
| Test | 2010 |

Dataset sizes:

- Training rows: 1941
- Calibration rows: 648
- Test rows: 341

## Results

### Point Prediction

| Metric | Value |
|---|---:|
| MAE | $16,871.76 |
| RMSE | $24,855.96 |

### Prediction Intervals

| Metric | Value |
|---|---:|
| Nominal coverage | 90% |
| Empirical coverage | 91.20% |
| Mean interval width | $73,428.89 |
| Conformal radius | $36,714.44 |

## Comparison With Primary Random Split

| Protocol | MAE | RMSE | Coverage | Width |
|---|---:|---:|---:|---:|
| Random split | $15,957.31 | $33,434.09 | 91.47% | $65,232.67 |
| Temporal split | $16,871.76 | $24,855.96 | 91.20% | $73,428.89 |

## Interpretation

The temporal evaluation shows that the frozen system maintains
prediction quality when evaluated on future-year observations.

The empirical coverage remains close to the nominal 90% target,
indicating that the conformal uncertainty intervals remain calibrated
under this temporal split.

The wider prediction intervals under temporal evaluation indicate
increased uncertainty when predicting future observations.

The experiment does not suggest modifying the primary model. It
provides evidence that the existing uncertainty pipeline is reasonably
robust to temporal distribution shift.

## Limitations

- Only one temporal split was evaluated.
- No model retraining or tuning was performed.
- Results should not be interpreted as proof of performance on all
future market conditions.