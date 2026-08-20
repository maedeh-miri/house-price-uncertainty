# Experiment 009 — Temporal Stress Test

## Question

Does the frozen ElasticNet + split conformal pipeline maintain
prediction accuracy and empirical interval coverage under temporal
distribution shift?

## Frozen configuration

Model:
- ElasticNet
- alpha = 0.1
- l1_ratio = 0.9

Conformal:
- split conformal
- absolute residual score
- nominal coverage = 0.90
- symmetric interval

## Temporal split

Train:
- 2006-2008

Calibration:
- 2009

Test:
- 2010

## Metrics

Point prediction:
- MAE
- RMSE

Intervals:
- empirical coverage
- covered count
- mean interval width

## Restrictions

This experiment does not modify:
- model hyperparameters
- feature schema
- preprocessing rules
- conformal method
- split boundaries

The purpose is evaluation under distribution shift, not model
improvement.