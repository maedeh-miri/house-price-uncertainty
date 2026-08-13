# Data

This project uses the full **Ames Housing dataset** for house-price
prediction and uncertainty evaluation.

Raw data is intentionally not committed to this repository.

## Dataset summary

- **Observations:** 2,930 residential property sales
- **Columns:** 82
- **Prediction target:** `SalePrice`
- **Subgroup variable:** `Neighborhood`
- **Raw filename:** `AmesHousing.txt`
- **File format:** tab-separated text
- **Missing target values:** 0

## Source

The dataset is obtained from the supplementary material accompanying the
Ames Housing study:

```text
https://jse.amstat.org/v19n3/decock/AmesHousing.txt
```

The raw dataset is not redistributed through this repository. Users should
download it from the original source and review the applicable usage and
redistribution terms.

## Citation

The Ames Housing dataset is described in:

> De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing
> Data as an End of Semester Regression Project*. Journal of Statistics
> Education, 19(3), 1–14.
> DOI: `10.1080/10691898.2011.11889627`

Users of the dataset should cite the original study when appropriate.

## Expected file location

Place the downloaded file at:

```text
data/raw/AmesHousing.txt
```

The `data/raw/` directory is excluded from Git through `.gitignore`.

## Download on Windows PowerShell

Run the following commands from the repository root:

```powershell
New-Item -ItemType Directory -Force data\raw
Invoke-WebRequest -Uri "https://jse.amstat.org/v19n3/decock/AmesHousing.txt" -OutFile "data\raw\AmesHousing.txt"
```

## File integrity

The version used during development has the following SHA256 checksum:

```text
6CFE6CB525BA437DE428653A1040E2AED7D696640BF75203786A6D7A0E67CFCC
```

Verify the downloaded file on Windows with:

```powershell
Get-FileHash data\raw\AmesHousing.txt -Algorithm SHA256
```

A different checksum may indicate that the source file has changed or that
the download is incomplete.

## Loading notes

The file is tab-separated and can be loaded with:

```python
import pandas as pd

data = pd.read_csv(
    "data/raw/AmesHousing.txt",
    sep="\t",
    keep_default_na=False,
    na_values=[""],
)
```

Using `keep_default_na=False` is intentional. Some categorical strings such
as `NA` represent structural absence of a property feature rather than an
unknown missing value.

The project therefore keeps structural absence separate from genuine
missingness. Learned imputation and preprocessing parameters will be fitted
using training data only and then applied without refitting to calibration
and test data.

## Initial validation results

The downloaded file was validated during project development:

- shape: 2,930 rows and 82 columns
- duplicate rows: 0
- duplicate column names: 0
- `SalePrice` is present
- `Neighborhood` is present
- missing `SalePrice` values: 0
- `SalePrice` data type: `int64`
- minimum `SalePrice`: 12,789
- maximum `SalePrice`: 755,000
- unique neighborhoods: 28
- file length: 2,931 lines, including the header row

## Completed audit and evaluation work

The initial file validation was followed by a structured data audit,
feature-availability review, and frozen evaluation protocol.

Completed work includes:

- missing-value analysis;
- structural-absence versus unknown-missing review;
- identifier and prediction-time leakage auditing;
- semantic and availability-based feature classification;
- a dedicated `Lot Frontage` imputation benchmark;
- a deterministic target-independent 60/20/20 evaluation split;
- a forward-looking temporal stress-test split;
- exact partition membership recorded in a committed manifest.

Detailed results are documented in:

- `reports/data_audit.md`
- `reports/feature_availability.md`
- `reports/feature_schema.csv`
- `reports/evaluation_protocol.md`
- `reports/evaluation_split_manifest.csv`

The full preprocessing strategy is not yet finalized. In particular,
imputation, categorical encoding, ordinal encoding, and scaling decisions
must be implemented in a leakage-safe pipeline and learned from training
data only.

## Licensing and redistribution

Project code is released under the MIT License. The MIT License applies to
the code in this repository and does not automatically apply to the dataset.

Dataset citation, usage, and redistribution remain subject to the terms of
the original source.