# Ames Housing Data Audit

## Scope

This audit examines the raw Ames Housing dataset before preprocessing
or model development. The raw source file is not modified.

The audit covers:

- source-file integrity
- target and schema validation
- missing-value patterns
- structural absence versus unknown values
- consistency checks for related feature groups
- robustness testing for `Lot Frontage` imputation

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
In many Ames Housing columns, these values represent structural absence
rather than unknown data.

Examples include:

- no garage
- no basement
- no pool
- no alley access
- no fence
- no fireplace

These categories must remain distinct from actual missing values.

For example:

- `Garage Type = NA` means that no garage is recorded
- `Garage Type = NaN` would mean that garage status is unknown
- `Mas Vnr Type = None` means that no masonry veneer is recorded
- `Mas Vnr Type = NaN` means that masonry-veneer information is unknown

## Garage consistency

- 157 properties are explicitly recorded as having no garage.
- 2 properties have a recorded garage type but a missing garage
  construction year.
- 1 of those properties also has missing garage capacity and area.
- no property marked as having no garage has a positive garage area or
  capacity.
- `Garage Type` contains no actual missing values.

The incomplete garage records represent unknown information rather than
structural absence. They will remain unchanged in the raw data and will
later be handled through training-only imputation and missing-value
indicators.

## Basement consistency

- 79 properties are explicitly recorded as having no basement.
- 1 property has an unknown basement status with the full basement feature
  block missing.
- 4 properties with a recorded basement contain an isolated missing
  categorical basement feature.
- no property marked as having no basement has a positive basement area.

Unknown basement information will not be converted into structural
absence without supporting evidence.

The audit therefore distinguishes three basement states:

1. basement absent
2. basement present
3. basement status unknown

## Masonry-veneer consistency

- 1,752 properties have `Mas Vnr Type = None`.
- 23 properties have both masonry type and area missing.
- 7 properties have type `None` but a positive masonry area.
- 3 properties have a recorded masonry type but an area of zero.
- no property has a recorded masonry type with a missing masonry area.

The 23 paired missing values are treated as unknown information rather
than confirmed absence.

The 10 type-area inconsistencies will remain unchanged in the raw data.
They will be documented and may later be represented by a derived
consistency indicator.

## Lot-frontage missingness

`Lot Frontage` contains 490 missing values, representing 16.72% of the
dataset.

Missingness is associated with observed property characteristics:

- missingness rates vary substantially across neighborhoods
- cul-de-sac properties have a higher missingness rate than inside lots
- properties with missing frontage have a larger median lot area
- neighborhood-level frontage medians vary considerably

Selected neighborhood missingness rates include:

| Neighborhood | Rows | Missing values | Missing percentage | Observed median frontage |
|---|---:|---:|---:|---:|
| `ClearCr` | 44 | 24 | 54.55% | 80.5 |
| `NWAmes` | 131 | 46 | 35.11% | 80.0 |
| `Sawyer` | 151 | 53 | 35.10% | 72.0 |
| `Gilbert` | 165 | 54 | 32.73% | 64.0 |
| `NridgHt` | 166 | 3 | 1.81% | 92.0 |

Two small neighborhoods contain no observed frontage values:

- `GrnHill`: 2 of 2 values missing
- `Landmrk`: 1 of 1 value missing

Missingness also varies by lot configuration:

| Lot configuration | Rows | Missing values | Missing percentage |
|---|---:|---:|---:|
| `CulDSac` | 180 | 88 | 48.89% |
| `FR3` | 14 | 4 | 28.57% |
| `FR2` | 85 | 20 | 23.53% |
| `Corner` | 511 | 104 | 20.35% |
| `Inside` | 2,140 | 274 | 12.80% |

Properties with missing frontage have a larger median lot area:

- frontage observed: 9,248.5
- frontage missing: 10,397.5

These patterns show that a single global median would ignore useful group
structure. They do not establish the exact mechanism that caused the
values to be missing.

## Lot-frontage imputation robustness experiment

Five candidate strategies were evaluated:

1. global median
2. neighborhood median
3. lot-configuration median
4. hierarchical neighborhood-and-lot-configuration median
5. multivariate histogram-gradient-boosting regression

The hierarchical strategy uses the following fallback sequence:

1. median for `Neighborhood` and `Lot Config`
2. neighborhood median
3. global training-set median

The model-based candidate uses property and land characteristics such as:

- `Lot Area`
- `Neighborhood`
- `Lot Config`
- `Lot Shape`
- `Land Contour`
- `MS Zoning`
- `Year Built`
- `Year Remod/Add`
- `Bldg Type`
- `House Style`
- `Overall Qual`
- `Overall Cond`
- `1st Flr SF`
- `Gr Liv Area`

`SalePrice` is deliberately excluded from the imputation model.

All preprocessing statistics and model parameters are learned from each
training split only.

### Repeated random cross-validation

Five-fold cross-validation was repeated ten times, producing 50
evaluations on the 2,440 properties with observed frontage values.

| Strategy | Mean MAE | MAE standard deviation | Mean RMSE | MAE wins |
|---|---:|---:|---:|---:|
| Global median | 16.650 | 0.637 | 23.330 | 0/50 |
| Lot-configuration median | 15.997 | 0.672 | 22.531 | 0/50 |
| Neighborhood median | 12.547 | 0.688 | 20.427 | 0/50 |
| Hierarchical median | 11.785 | 0.639 | 19.611 | 0/50 |
| Model-based HGB | **8.381** | **0.443** | **14.292** | **50/50** |

Relative to the global median, the model-based candidate reduced:

- mean MAE by approximately 49.7%
- mean RMSE by approximately 38.9%

Relative to the hierarchical median, the model-based candidate reduced:

- mean MAE by approximately 28.9%
- mean RMSE by approximately 27.1%

The hierarchical strategy required its primary fallback for approximately
0.50% of validation rows.

### Conditional masking

A second evaluation used 20 repeated masks weighted toward observed
`Neighborhood` and `Lot Config` patterns associated with natural
missingness.

Each repetition masked 408 observed frontage values, approximating the
dataset's natural missingness percentage.

| Strategy | Mean MAE | MAE standard deviation | Mean RMSE | MAE wins |
|---|---:|---:|---:|---:|
| Global median | 16.647 | 0.638 | 23.474 | 0/20 |
| Lot-configuration median | 15.664 | 0.611 | 22.483 | 0/20 |
| Neighborhood median | 12.882 | 0.608 | 20.814 | 0/20 |
| Hierarchical median | 11.822 | 0.625 | 19.952 | 0/20 |
| Model-based HGB | **9.229** | **0.408** | **15.360** | **20/20** |

Relative to the global median, the model-based candidate reduced:

- mean MAE by approximately 44.5%
- mean RMSE by approximately 34.6%

Relative to the hierarchical median, the model-based candidate reduced:

- mean MAE by approximately 21.9%
- mean RMSE by approximately 23.0%

The hierarchical strategy required its primary fallback for approximately
1.41% of conditionally masked rows.

### Interpretation and decision

The hierarchical neighborhood-and-lot-configuration median is retained
as the strongest simple and interpretable baseline.

The model-based HGB imputer is retained as the leading reconstruction
candidate because it:

- achieved the lowest MAE and RMSE under both evaluation protocols
- won all 50 repeated-cross-validation evaluations
- won all 20 conditional-masking evaluations
- showed lower MAE variability than the median-based candidates

The model-based candidate is not yet declared the final production
imputer. Reconstruction accuracy for `Lot Frontage` does not guarantee
better downstream house-price predictions.

The hierarchical baseline and model-based candidate must therefore be
compared inside the final house-price pipeline.

The final selection will consider:

- house-price MAE and RMSE
- prediction-interval coverage
- prediction-interval width
- subgroup coverage by neighborhood
- implementation complexity
- training and inference cost

### Experiment limitations

The experiment evaluates artificially hidden values for which ground
truth is available. Naturally missing frontage values may follow a
different distribution.

Conditional masking approximates observed missingness patterns but does
not recreate the true missing-data mechanism.

Groups with no observed frontage values, including `GrnHill` and
`Landmrk`, cannot be directly validated because no ground-truth frontage
values exist for them.

The experiment therefore identifies the strongest candidates for
downstream evaluation rather than proving the true values of the 490
naturally missing observations.

### Reproducibility artifacts

The experiment is recorded in:

- `experiments/lot_frontage_imputation_benchmark.py`
- `experiments/experiment_001_notes.md`
- `experiments/results/lot_frontage_imputation_detailed.csv`
- `experiments/results/lot_frontage_imputation_summary.csv`
- `experiments/results/lot_frontage_imputation_metadata.json`

## Preprocessing principles

1. Raw values will not be manually overwritten.
2. Structural absence and unknown values will remain distinct.
3. Imputation parameters will be learned from training data only.
4. Missing-value indicators will be considered for informative missingness.
5. Consistency flags may be derived without modifying the source columns.
6. Preprocessing will be implemented inside leakage-safe pipelines.
7. Imputation candidates will be evaluated through downstream model
   performance rather than reconstruction error alone.

## Remaining audit work

- review identifier columns
- identify leakage-prone and post-sale variables
- inspect rare categorical levels
- define train, calibration, and test splits
- compare the leading frontage imputers inside the downstream price model
- document the final evaluation protocol
