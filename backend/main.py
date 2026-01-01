from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# -----------------------
# APP INIT
# -----------------------
app = FastAPI(title="Breast Cancer Prediction API")

# -----------------------
# CORS 
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# PATHS
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# -----------------------
# LOAD MODELS
# -----------------------
try:
    lr_model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    dt_model = joblib.load(MODEL_DIR / "decision_tree.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

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

# -----------------------
# SERVE STATIC FILES
# -----------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    """Serve the index.html"""
    return FileResponse(STATIC_DIR / "index.html")

# -----------------------
# API ROUTES
# -----------------------
@app.post("/predict/logistic")
def predict_logistic(data: CancerInput):
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {EXPECTED_FEATURES} features"
        )

    X = pd.DataFrame([data.features], columns=FEATURE_NAMES)
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

    X = pd.DataFrame([data.features], columns=FEATURE_NAMES)
    prediction = int(dt_model.predict(X)[0])

    return {
        "model": "Decision Tree",
        "prediction": prediction
    }
