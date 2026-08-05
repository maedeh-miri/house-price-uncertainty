# Feature Availability and Leakage Audit

## Scope

This audit determines whether candidate predictors are appropriate for
the intended prediction scenario.

A feature is not classified as leakage solely because it is strongly
associated with the target. Its validity depends on its meaning,
stability, and availability at prediction time.

## Prediction contract

The primary model estimates a property's expected market value before
the sale transaction is finalized.

The primary predictors must represent property information that is
available during pre-sale valuation.

Variables describing the realized sale transaction are excluded from the
primary model because their availability is not guaranteed at valuation
time.

## Feature decisions

| Column | Role | Primary-model decision | Rationale |
|---|---|---|---|
| `SalePrice` | prediction target | exclude from predictors | This is the value being predicted. |
| `Order` | observation identifier | exclude | Unique monotonic sequence with no property meaning. |
| `PID` | parcel identifier | exclude from predictors | Opaque identifier that may act as an unstable location proxy. |
| `Neighborhood` | property location | keep as categorical | Interpretable property information available before sale. |
| `MS SubClass` | coded property class | keep as categorical | Numeric values represent categories rather than continuous measurements. |
| `Mo Sold` | realized transaction timing | exclude from primary model | The actual future sale month may not be known during valuation. |
| `Yr Sold` | realized transaction timing | exclude from primary model | The actual future sale year may not be known during valuation. |
| `Sale Type` | transaction context | exclude from primary model | The finalized transaction type may not be available during valuation. |
| `Sale Condition` | transaction context | exclude from primary model | The finalized or unusual sale condition may not yet be known. |

## Identifiers

### `Order`

`Order` is unique for every observation and increases monotonically from
1 to 2,930.

It functions as a row number rather than a property characteristic.

It may be retained for debugging or report references but must not enter
the model feature matrix.

### `PID`

`PID` is unique for every property and identifies the parcel.

Although its correlation with `SalePrice` is approximately -0.247, this
does not justify using it as a predictor.

The identifier may indirectly encode location or administrative ordering,
but such information is opaque, difficult to interpret, and potentially
unstable outside the source dataset.

`Neighborhood` provides an explicit and interpretable representation of
location and is preferred over an identifier proxy.

`PID` may be retained as metadata for:

- tracing predictions to source records
- investigating large prediction errors
- reporting problematic observations
- joining predictions with external records

## Coded categorical variables

`MS SubClass` contains numeric codes but does not represent a continuous
measurement.

For example, class 120 is not meaningfully twice class 60.

The column must therefore be processed as categorical data rather than
as an ordinary numeric predictor.

## Transaction-variable findings

Sale prices vary substantially across transaction categories.

Examples include:

- median `Sale Type = New`: 250,580
- median `Sale Type = WD`: 157,000
- median `Sale Condition = Partial`: 250,000
- median `Sale Condition = Abnorml`: 129,450

These differences demonstrate predictive association. They do not prove
that the transaction variables are causally responsible for the price
differences or valid for the primary prediction scenario.

New properties may also differ in age, quality, size, and other physical
characteristics.

## Redundancy between transaction variables

All 239 observations with `Sale Type = New` also have
`Sale Condition = Partial`.

These observations account for 239 of the dataset's 245 partial sales.

`Sale Type` and `Sale Condition` therefore contain strongly overlapping
information for new-construction transactions in this dataset.

## Time variables

Median sale prices vary across sale months and years.

The primary reason for excluding `Mo Sold` and `Yr Sold` is not weak
association. It is the absence of a guaranteed match between the
property-valuation date and the realized future sale date.

These variables may be valid in a different prediction contract where
the model is used immediately before a known transaction date.

## Model variants

### Primary property model

The primary model excludes:

- `Order`
- `PID`
- `Mo Sold`
- `Yr Sold`
- `Sale Type`
- `Sale Condition`

It retains property characteristics such as `Neighborhood` and treats
`MS SubClass` as categorical.

### Market-aware sensitivity model

A market-aware sensitivity model may add:

- `Mo Sold`
- `Yr Sold`

This model assumes that prediction occurs close enough to the transaction
for its timing to be known.

Evaluation of this model should respect the temporal structure of the
data.

### Transaction-aware sensitivity model

A separate sensitivity model may additionally include:

- `Sale Type`
- `Sale Condition`

This model will quantify how much realized transaction context changes
predictive performance.

It is not treated as the primary pre-sale model.

### Normal-sale sensitivity cohort

A later analysis may evaluate the property model on observations with:

```text
Sale Condition = Normal
```

This will test whether performance and uncertainty differ when unusual
transactions are excluded.

## Current primary-model exclusions

- `SalePrice` from the predictor matrix
- `Order`
- `PID`
- `Mo Sold`
- `Yr Sold`
- `Sale Type`
- `Sale Condition`

## Leakage controls to implement

- define the predictor schema explicitly
- prevent identifier columns from entering preprocessing
- prevent the target from entering feature transformations
- cast `MS SubClass` to a categorical representation
- test that excluded transaction columns are absent from the primary model
- fit all preprocessing operations on training data only
- evaluate transaction-aware variants separately from the primary model

## Remaining work

- audit all remaining columns for prediction-time availability
- inspect rare categorical levels
- define the final feature schema
- implement automated schema and leakage tests
- define train, calibration, and test splits
- document the final evaluation protocol
