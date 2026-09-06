import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from evidently import Report
from evidently.presets import DataDriftPreset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

OUTPUT_DATASET = os.path.join(BASE_DIR, "prediction_data_100.csv")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "drift_summary.csv")
OUTPUT_REPORT = os.path.join(BASE_DIR, "drift_report.html")


RANDOM_STATE = 42
N_ROWS = 100


def generate_prediction_dataset(training_data):
    """Generate 100 random input rows using the training feature distributions."""

    rng = np.random.default_rng(RANDOM_STATE)

    feature_columns = [
        "age",
        "gender",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    generated = pd.DataFrame()

    categorical_columns = [
        "gender",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    numerical_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    # Generate categorical values according to the training distribution.
    for column in categorical_columns:
        values = training_data[column].dropna().values
        probabilities = (
            training_data[column]
            .dropna()
            .value_counts(normalize=True)
            .reindex(pd.unique(values))
            .values
        )

        generated[column] = rng.choice(
            pd.unique(values),
            size=N_ROWS,
            p=probabilities,
        )

    # Generate numerical values uniformly within the training range.
    for column in numerical_columns:
        values = pd.to_numeric(training_data[column], errors="coerce").dropna()

        generated[column] = rng.uniform(
            values.min(),
            values.max(),
            size=N_ROWS,
        )

    return generated[feature_columns]


def calculate_drift_summary(training_data, prediction_data):
    """Calculate KS-statistic based drift summary for numeric features."""

    rows = []

    numeric_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    for column in numeric_columns:
        reference = pd.to_numeric(
            training_data[column], errors="coerce"
        ).dropna()

        current = pd.to_numeric(
            prediction_data[column], errors="coerce"
        ).dropna()

        statistic, p_value = ks_2samp(reference, current)

        rows.append(
            {
                "feature": column,
                "test": "Kolmogorov-Smirnov",
                "statistic": round(float(statistic), 6),
                "p_value": round(float(p_value), 6),
                "drift_detected": bool(p_value < 0.05),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("TASK 7 - INPUT DRIFT DETECTION")
    print("=" * 70)

    # Load training data.
    data = pd.read_csv(DATA_PATH)

    # Remove observation ID and target.
    training_data = data.drop(columns=["sno", "target"]).copy()

    print(f"Training dataset shape: {training_data.shape}")

    # Generate 100-row prediction dataset.
    prediction_data = generate_prediction_dataset(training_data)

    prediction_data.to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    print(f"Generated prediction dataset: {prediction_data.shape}")
    print(f"Saved to: {OUTPUT_DATASET}")

    # Calculate statistical drift summary.
    drift_summary = calculate_drift_summary(
        training_data,
        prediction_data,
    )

    drift_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print("\nNumerical feature drift summary:")
    print(drift_summary.to_string(index=False))

    # Generate Evidently data-drift report.
    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    result = report.run(
        current_data=prediction_data,
        reference_data=training_data,
    )

    result.save_html(OUTPUT_REPORT)

    print(f"\nEvidently report saved to: {OUTPUT_REPORT}")

    print("\nOutput files created:")
    print(f"1. {OUTPUT_DATASET}")
    print(f"2. {OUTPUT_SUMMARY}")
    print(f"3. {OUTPUT_REPORT}")

    print("\nTask 7 Stage 1 completed.")


if __name__ == "__main__":
    main()
