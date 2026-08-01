# Data

Raw data is intentionally not committed to the repository.

## Required next step

Document the original dataset before training code is added:

- dataset name and source
- license and redistribution terms
- target column
- row and feature counts
- missing-value patterns
- duplicate checks
- potential leakage columns
- train/test split policy

Place local raw files under `data/raw/`. That directory is ignored by Git.
