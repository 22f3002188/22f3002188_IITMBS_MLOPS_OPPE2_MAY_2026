import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("data.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. Prepare features and target
# ============================================================

# sno is only an observation index and is not a clinical feature.
X = df.drop(columns=["target", "sno"])

# Convert target to binary.
y = df["target"].map({"no": 0, "yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in target column.")


# ============================================================
# 3. Create age groups
# ============================================================

# Age is the sensitive attribute required by the OPPE question.
age_group = pd.cut(
    df["age"],
    bins=[0, 49, 59, np.inf],
    labels=["<50", "50-59", "60+"],
    include_lowest=True
)

print("\nAge-group distribution:")
print(age_group.value_counts().sort_index())


# ============================================================
# 4. Preprocessing
# ============================================================

categorical_features = ["gender"]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="if_binary",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)


# ============================================================
# 5. Train/test split
# ============================================================

X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
    X,
    y,
    age_group,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. Transform data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 7. Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_processed, y_train)


# ============================================================
# 8. Predictions
# ============================================================

y_pred = model.predict(X_test_processed)


# ============================================================
# 9. Basic performance
# ============================================================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    f"Overall accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}"
)


# ============================================================
# 10. Fairlearn MetricFrame
# ============================================================

metrics = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_test
)

print("\n==============================")
print("FAIRLEARN GROUP METRICS")
print("==============================")

print(metric_frame.by_group)


# ============================================================
# 11. Additional fairness metrics
# ============================================================

dp_difference = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_test
)

eo_difference = equalized_odds_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_test
)

print("\n==============================")
print("FAIRNESS METRICS")
print("==============================")

print(
    f"Demographic Parity Difference: "
    f"{dp_difference:.4f}"
)

print(
    f"Equalized Odds Difference: "
    f"{eo_difference:.4f}"
)


# ============================================================
# 12. Save results
# ============================================================

group_results = metric_frame.by_group.reset_index()

group_results.to_csv(
    "fairlearn_age_group_metrics.csv",
    index=False
)

summary = pd.DataFrame(
    {
        "metric": [
            "overall_accuracy",
            "demographic_parity_difference",
            "equalized_odds_difference",
        ],
        "value": [
            accuracy_score(y_test, y_pred),
            dp_difference,
            eo_difference,
        ],
    }
)

summary.to_csv(
    "fairness_summary.csv",
    index=False
)


# ============================================================
# 13. Plain-English interpretation
# ============================================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")

print(
    "Fairness was evaluated using age as the sensitive attribute, "
    "with patients divided into three age groups: <50, 50-59, and 60+."
)

print(
    "Demographic Parity Difference and Equalized Odds Difference "
    "measure disparities in model predictions and error-related "
    "behavior between the age groups."
)

print(
    "Values closer to 0 indicate smaller differences between "
    "the age groups."
)

print("\nSaved:")
print("  fairlearn_age_group_metrics.csv")
print("  fairness_summary.csv")
