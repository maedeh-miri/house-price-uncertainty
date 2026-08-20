# Experiment 010 — Conformal Sensitivity Analysis

## Status

This experiment evaluates how the conformal prediction system changes under different nominal coverage targets.

This is a sensitivity analysis only.

The experiment does not modify:
- the point prediction model;
- preprocessing;
- train/calibration/test splits;
- feature selection;
- primary evaluation protocol.

The frozen 90% conformal configuration remains the primary uncertainty result.

## Objective

The goal is to measure the trade-off between:
- empirical coverage;
- prediction interval width;
- uncertainty sharpness.

The evaluated nominal coverage levels are:
- 80%
- 90%
- 95%

## Frozen Predictive System

Point model:
- Model: ElasticNet
- alpha: 0.1
- l1_ratio: 0.9
- Target: SalePrice
- Target transformation: none

Conformal prediction:
- Method: split conformal prediction
- Score: absolute residual
- Interval type: symmetric

## Evaluation Protocol

The analysis uses the frozen primary random split.

Dataset partitions:
- Training rows: 1758
- Calibration rows: 586
- Test rows: 586

The calibration partition remains unchanged for all sensitivity levels.

For each nominal coverage target:

1. Train the point model on the frozen training partition.
2. Compute calibration residuals.
3. Calculate the conformal quantile.
4. Generate prediction intervals on the untouched test partition.
5. Measure empirical coverage and interval width.

## Evaluated Coverage Levels

| Configuration | Target Coverage |
|---|---:|
| Low uncertainty | 80% |
| Primary setting | 90% |
| Conservative setting | 95% |

## Expected Behavior

Lower coverage targets:
- produce narrower intervals;
- reduce uncertainty width;
- allow more missed observations.

Higher coverage targets:
- produce wider intervals;
- increase uncertainty width;
- reduce missed observations.

## Outputs

The experiment should produce:

experiments/results/conformal_sensitivity_summary.json

The output should include:
- nominal coverage;
- conformal radius;
- empirical coverage;
- mean interval width;
- covered observations;
- total observations.

## Interpretation Rules

The 90% conformal configuration remains the primary uncertainty result.

The 80% and 95% configurations are sensitivity analyses only.

No alternative coverage level will replace the frozen primary result.

The purpose of this experiment is to understand the reliability-width trade-off of the conformal uncertainty system.