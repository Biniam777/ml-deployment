from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Breast Cancer Prediction API")

# -----------------------
# CORS (REQUIRED)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# LOAD MODELS
# -----------------------
lr_model = joblib.load("models/logistic_regression.joblib")
dt_model = joblib.load("models/decision_tree.joblib")
scaler = joblib.load("models/scaler.joblib")

# -----------------------
# FEATURE CONFIG
# -----------------------
FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]

EXPECTED_FEATURES = len(FEATURE_NAMES)


class CancerInput(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API is running"}


@app.post("/predict/logistic")
def predict_logistic(data: CancerInput):
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {EXPECTED_FEATURES} features"
        )

    # Convert input to DataFrame with feature names
    X = pd.DataFrame([data.features], columns=FEATURE_NAMES)

    # Scale & predict
    X_scaled = scaler.transform(X)
    prediction = int(lr_model.predict(X_scaled)[0])

    return {
        "model": "Logistic Regression",
        "prediction": prediction
    }


@app.post("/predict/tree")
def predict_tree(data: CancerInput):
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {EXPECTED_FEATURES} features"
        )

    # Convert input to DataFrame with feature names
    X = pd.DataFrame([data.features], columns=FEATURE_NAMES)

    prediction = int(dt_model.predict(X)[0])

    return {
        "model": "Decision Tree",
        "prediction": prediction
    }
