import os
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Logging configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("heart-disease-api")


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")


# ============================================================
# Load model
# ============================================================

try:
    model_package = joblib.load(MODEL_PATH)

    preprocessor = model_package["preprocessor"]
    model = model_package["model"]
    FEATURES = model_package["features"]

    logger.info("Model loaded successfully")

except Exception as exc:
    logger.exception("Failed to load model")
    raise RuntimeError(
        f"Could not load model from {MODEL_PATH}"
    ) from exc


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production-style API for heart disease prediction",
    version="1.0.0"
)


# ============================================================
# Request schema
# ============================================================

class PatientData(BaseModel):
    age: float
    gender: str
    cp: float
    trestbps: float | None = None
    chol: float | None = None
    fbs: float | None = None
    restecg: float | None = None
    thalach: float | None = None
    exang: float | None = None
    oldpeak: float | None = None
    slope: float | None = None
    ca: float | None = None
    thal: float | None = None


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    try:
        # Convert request to DataFrame.
        input_data = patient.model_dump()

        df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        # Apply exactly the same preprocessing used during training.
        processed_data = preprocessor.transform(df)

        # Prediction.
        prediction = int(
            model.predict(processed_data)[0]
        )

        probabilities = model.predict_proba(
            processed_data
        )[0]

        probability = float(
            probabilities[1]
        )

        prediction_label = (
            "yes"
            if prediction == 1
            else "no"
        )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        # Individual prediction log.
        logger.info(
            "prediction_request | "
            "timestamp=%s | "
            "input=%s | "
            "prediction=%s | "
            "probability=%.6f",
            timestamp,
            input_data,
            prediction_label,
            probability
        )

        return {
            "prediction": prediction_label,
            "heart_disease_probability": round(
                probability,
                6
            ),
            "timestamp": timestamp
        }

    except Exception as exc:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc
