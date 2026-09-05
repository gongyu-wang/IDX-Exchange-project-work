"""Train the compact, deployable model used by the Streamlit application."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "ClosePrice"
DATE_COLUMN = "CloseDate"
NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
]
CATEGORICAL_FEATURES = ["City", "CountyOrParish", "PostalCode"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Cleaned CRMLS CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "app_model.joblib",
        help="Destination joblib artifact",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=150_000,
        help="Optional deterministic cap for practical local training",
    )
    return parser.parse_args()


def load_training_frame(path: Path) -> pd.DataFrame:
    required = FEATURES + [TARGET, DATE_COLUMN]
    frame = pd.read_csv(path, usecols=lambda column: column in required, low_memory=False)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"The data file is missing required columns: {', '.join(missing)}")

    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame["PostalCode"] = frame["PostalCode"].astype("string").str.replace(r"\.0$", "", regex=True)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object").where(frame[column].notna(), np.nan)

    frame = frame.dropna(subset=[TARGET, DATE_COLUMN, "LivingArea"])
    frame = frame[(frame[TARGET] > 0) & (frame["LivingArea"] > 0)].copy()
    if frame.empty:
        raise ValueError("No usable rows remain after validating target, date, and living area.")
    return frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    months = frame[DATE_COLUMN].dt.to_period("M")
    test_month = months.max()
    train = frame.loc[months < test_month].copy()
    test = frame.loc[months == test_month].copy()
    if train.empty or test.empty:
        raise ValueError("At least two distinct CloseDate months are required for training.")
    return train, test, str(test_month)


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=24,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def main() -> None:
    args = parse_args()
    frame = load_training_frame(args.data)
    train, test, test_month = chronological_split(frame)

    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=42).sort_values(DATE_COLUMN)

    pipeline = build_pipeline()
    pipeline.fit(train[FEATURES], train[TARGET])
    predictions = pipeline.predict(test[FEATURES])
    percentage_errors = np.abs((test[TARGET].to_numpy() - predictions) / test[TARGET].to_numpy())

    artifact = {
        "pipeline": pipeline,
        "metrics": {
            "r2": float(r2_score(test[TARGET], predictions)),
            "mae": float(mean_absolute_error(test[TARGET], predictions)),
            "mape": float(percentage_errors.mean() * 100),
            "mdape": float(np.median(percentage_errors) * 100),
        },
        "metadata": {
            "model_name": "Random Forest deployment model",
            "features": FEATURES,
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_month": test_month,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"Saved model artifact to {args.output}")
    print(pd.Series(artifact["metrics"]).to_string())


if __name__ == "__main__":
    main()
