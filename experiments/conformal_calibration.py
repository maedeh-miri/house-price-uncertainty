"""Calibrate the frozen symmetric split-conformal prediction interval."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from house_price_uncertainty.conformal import (
    calibrate_symmetric_conformal,
)
from house_price_uncertainty.data import load_ames_housing
from house_price_uncertainty.feature_schema import (
    prepare_primary_model_data,
)
from house_price_uncertainty.models import (
    build_elasticnet_pipeline,
)
from house_price_uncertainty.splitting import (
    make_random_evaluation_split,
)

EXPERIMENT_ID = "experiment_007"
PROTOCOL_NAME = "primary_random"

EXPECTED_TRAIN_ROWS = 1_758
EXPECTED_CALIBRATION_ROWS = 586

POINT_MODEL_NAME = "elasticnet"
ELASTICNET_ALPHA = 0.1
ELASTICNET_L1_RATIO = 0.9

TARGET_NAME = "SalePrice"
TARGET_SCALE = "raw_sale_price"

NOMINAL_COVERAGE = 0.90
EXPECTED_QUANTILE_RANK = 529

OUTPUT_PATH = Path(
    "experiments/results/conformal_calibration_summary.json"
)


def main() -> None:
    """Fit the frozen point model and calibrate its conformal radius."""
    data = load_ames_housing()

    split = make_random_evaluation_split(
        data,
        first_random_state=42,
        second_random_state=43,
    )

    if len(split.train) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected training partition size: "
            f"expected {EXPECTED_TRAIN_ROWS}, got {len(split.train)}."
        )

    if len(split.calibration) != EXPECTED_CALIBRATION_ROWS:
        raise RuntimeError(
            "Unexpected calibration partition size: "
            f"expected {EXPECTED_CALIBRATION_ROWS}, "
            f"got {len(split.calibration)}."
        )

    train = prepare_primary_model_data(
        split.train,
    )
    calibration = prepare_primary_model_data(
        split.calibration,
    )

    model = build_elasticnet_pipeline(
        alpha=ELASTICNET_ALPHA,
        l1_ratio=ELASTICNET_L1_RATIO,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            train.features,
            train.target,
        )

    convergence_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    if convergence_warnings:
        raise RuntimeError(
            "ElasticNet emitted a convergence warning during the "
            "full-training fit."
        )

    calibration_predictions = model.predict(
        calibration.features,
    )

    conformal_calibration = calibrate_symmetric_conformal(
        calibration.target,
        calibration_predictions,
        coverage=NOMINAL_COVERAGE,
    )

    if (
        conformal_calibration.quantile_rank
        != EXPECTED_QUANTILE_RANK
    ):
        raise RuntimeError(
            "Unexpected conformal quantile rank: "
            f"expected {EXPECTED_QUANTILE_RANK}, "
            f"got {conformal_calibration.quantile_rank}."
        )

    absolute_residuals = np.abs(
        np.asarray(
            calibration.target,
            dtype=float,
        )
        - np.asarray(
            calibration_predictions,
            dtype=float,
        )
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL_NAME,
        "point_model": {
            "name": POINT_MODEL_NAME,
            "alpha": ELASTICNET_ALPHA,
            "l1_ratio": ELASTICNET_L1_RATIO,
            "target": TARGET_NAME,
            "target_scale": TARGET_SCALE,
            "target_transform": "none",
        },
        "data": {
            "train_rows": len(train.target),
            "calibration_rows": len(calibration.target),
        },
        "conformal": {
            "method": "split_conformal",
            "score": "absolute_residual",
            "interval_type": "symmetric",
            "nominal_coverage": (
                conformal_calibration.coverage
            ),
            "quantile_rank": (
                conformal_calibration.quantile_rank
            ),
            "radius": (
                conformal_calibration.radius
            ),
            "interval_width": (
                2.0 * conformal_calibration.radius
            ),
            "clip_lower_bound_at_zero": False,
        },
        "calibration_score_summary": {
            "minimum": float(
                np.min(absolute_residuals)
            ),
            "median": float(
                np.median(absolute_residuals)
            ),
            "mean": float(
                np.mean(absolute_residuals)
            ),
            "maximum": float(
                np.max(absolute_residuals)
            ),
        },
        "test_evaluated": False,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Training rows: {len(train.target)}"
    )
    print(
        f"Calibration rows: {len(calibration.target)}"
    )
    print(
        "Nominal coverage: "
        f"{conformal_calibration.coverage:.2%}"
    )
    print(
        "Finite-sample quantile rank: "
        f"{conformal_calibration.quantile_rank}"
    )
    print(
        "Conformal radius: "
        f"${conformal_calibration.radius:,.2f}"
    )
    print(
        "Symmetric interval width: "
        f"${2.0 * conformal_calibration.radius:,.2f}"
    )
    print(
        f"Saved summary: {OUTPUT_PATH}"
    )
    print(
        "Test partition evaluated: False"
    )


if __name__ == "__main__":
    main()