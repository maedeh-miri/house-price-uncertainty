# Experiments

This directory contains reproducible analyses used to make or audit
modeling decisions.

A completed experiment should normally include:

- an executable experiment script;
- a short notes file describing the question and interpretation;
- machine-readable outputs under `experiments/results/`;
- random seeds and relevant configuration or metadata;
- enough information to reproduce the reported result.

The exact output format depends on the experiment. Results may use
CSV, JSON, or both.

Reusable model configurations belong in `configs/`.

Experiment notes should record:

- the question or hypothesis;
- the data and evaluation protocol;
- important implementation choices;
- results;
- interpretation;
- limitations;
- negative or inconclusive findings.

Generated results should not be manually edited when they can be
reproduced by the corresponding experiment script.
