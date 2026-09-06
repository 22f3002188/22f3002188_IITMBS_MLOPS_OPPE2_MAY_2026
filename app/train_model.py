import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

# sno is an observation index, not a predictive feature.
X = df.drop(columns=["target", "sno"])

y = df["target"].map({
    "no": 0,
    "yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected values found in target column.")


# --------------------------------------------------
# Feature types
# --------------------------------------------------

categorical_features = ["gender"]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
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
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    verbose_feature_names_out=False
)


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Preprocess
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# Train model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

predictions = model.predict(X_test_processed)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# Save preprocessing + model together
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_package = {
    "preprocessor": preprocessor,
    "model": model,
    "features": list(X.columns),
}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(MODEL_PATH)
