# Ames Housing Data Audit

## Scope

This audit examines the raw Ames Housing dataset before preprocessing
or model development. The raw source file is not modified.

## Source-file validation

- rows: 2,930
- columns: 82
- duplicate rows: 0
- duplicate column names: 0
- missing `SalePrice` values: 0
- `SalePrice` type: `int64`
- minimum `SalePrice`: 12,789
- maximum `SalePrice`: 755,000
- unique neighborhoods: 28
- verified SHA256:
  `6CFE6CB525BA437DE428653A1040E2AED7D696640BF75203786A6D7A0E67CFCC`

## Missing-value summary

- columns containing actual missing values: 21
- total missing cells: 719
- rows containing at least one missing value: 661
- maximum missing values in one row: 11

The largest missing-value counts are:

| Column | Missing count | Missing percentage |
|---|---:|---:|
| `Lot Frontage` | 490 | 16.72% |
| `Garage Yr Blt` | 159 | 5.43% |
| `Mas Vnr Type` | 23 | 0.78% |
| `Mas Vnr Area` | 23 | 0.78% |

## Structural absence versus unknown values

Literal categorical values such as `NA` are preserved during loading.
They frequently represent structural absence rather than unknown data.

Examples include:

- no garage
- no basement
- no pool
- no alley access
- no fence
- no fireplace

These values must not be treated as ordinary missing values.

## Garage consistency

- 157 properties are explicitly recorded as having no garage.
- 2 properties have a garage type but a missing garage construction year.
- 1 of those properties also has missing garage capacity and area.
- no property marked as having no garage has a positive garage area or
  capacity.

The incomplete garage records will be preserved and handled through
training-only imputation and missing-value indicators.

## Basement consistency

- 79 properties are explicitly recorded as having no basement.
- 1 property has an unknown basement status with the full basement block
  missing.
- 4 properties with a recorded basement contain an isolated missing
  categorical basement feature.
- no property marked as having no basement has a positive basement area.

Unknown basement information will not be converted into structural
absence without supporting evidence.

## Masonry-veneer consistency

- 1,752 properties have `Mas Vnr Type = None`.
- 23 properties have both masonry type and area missing.
- 7 properties have type `None` but a positive masonry area.
- 3 properties have a recorded masonry type but an area of zero.
- no property has a recorded masonry type with a missing masonry area.

The 23 paired missing values are treated as unknown information rather
than confirmed absence.

The 10 type-area inconsistencies will remain unchanged in the raw data.
They will be documented and may be represented by a derived consistency
indicator during preprocessing.

## Preprocessing principles

1. Raw values will not be manually overwritten.
2. Structural absence and unknown values will remain distinct.
3. Imputation parameters will be learned from training data only.
4. Missing-value indicators will be considered for informative missingness.
5. Consistency flags may be derived without modifying the source columns.
6. Preprocessing will be implemented inside leakage-safe pipelines.

## Remaining audit work

- analyze missing-value patterns for `Lot Frontage`
- review identifier columns
- identify leakage-prone and post-sale variables
- inspect rare categorical levels
- define train, calibration, and test splits
- document the final evaluation protocol