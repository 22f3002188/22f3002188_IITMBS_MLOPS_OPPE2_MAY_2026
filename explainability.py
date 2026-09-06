import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("data.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 2. Prepare X and y
# sno is only an observation index, so exclude it.
# --------------------------------------------------
X = df.drop(columns=["target", "sno"])
y = df["target"].map({"no": 0, "yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in target column.")


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------
categorical_features = ["gender"]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------
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


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Preprocess data
# --------------------------------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()


# --------------------------------------------------
# 7. Train Random Forest
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_processed, y_train)


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------
predictions = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# --------------------------------------------------
# 9. SHAP explainability
# --------------------------------------------------
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test_processed)

# SHAP versions may return either:
# (samples, features, classes)
# or a list containing one matrix per class.
if isinstance(shap_values, list):
    values = shap_values[1]
elif shap_values.ndim == 3:
    values = shap_values[:, :, 1]
else:
    values = shap_values


# --------------------------------------------------
# 10. Calculate global feature importance
# --------------------------------------------------
mean_abs_shap = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }
).sort_values(
    "mean_abs_shap",
    ascending=True
)

print("\n==============================")
print("SHAP FEATURE IMPORTANCE")
print("LOWEST IMPACT -> HIGHEST IMPACT")
print("==============================")

print(importance_df.to_string(index=False))


# --------------------------------------------------
# 11. Save numerical results
# --------------------------------------------------
importance_df.to_csv(
    "shap_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# 12. Save SHAP summary plot
# --------------------------------------------------
shap.summary_plot(
    values,
    X_test_processed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 13. Plain-English least-impact result
# --------------------------------------------------
least_impact = importance_df.head(3)

print("\n==============================")
print("LEAST IMPACT FEATURES")
print("==============================")

for _, row in least_impact.iterrows():
    print(
        f"{row['feature']}: "
        f"mean absolute SHAP = {row['mean_abs_shap']:.6f}"
    )

print("\nSaved:")
print("  shap_feature_importance.csv")
print("  shap_summary.png")
