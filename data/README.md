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

Invoke-WebRequest `
  -Uri "https://jse.amstat.org/v19n3/decock/AmesHousing.txt" `
  -OutFile "data\raw\AmesHousing.txt"
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
as `NA` may represent the absence of a property feature rather than an
unknown value. Missing-value handling will therefore be defined explicitly
in the preprocessing pipeline.

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

The next validation stage will examine:

- missing-value patterns by column
- identifier columns
- suspicious post-sale or leakage-prone variables
- rare categorical levels
- inconsistent or impossible feature values
- reproducible train, calibration, and test splits

## Licensing and redistribution

Project code is released under the MIT License. The MIT License applies to
the code in this repository and does not automatically apply to the dataset.

Dataset citation, usage, and redistribution remain subject to the terms of
the original source.