# ============================================================
# app.py — Health Risk Analyzer API
# ============================================================

import os
import pickle
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Health Risk Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR     = Path(__file__).resolve().parent          # .../HealthRisk-Analyzer/api
PROJECT_DIR  = BASE_DIR.parent                          # .../HealthRisk-Analyzer
MODELS_DIR   = PROJECT_DIR / "models"
FRONTEND_DIR = PROJECT_DIR / "frontend"

print(f"📁 BASE_DIR     : {BASE_DIR}")
print(f"📁 PROJECT_DIR  : {PROJECT_DIR}")
print(f"📁 MODELS_DIR   : {MODELS_DIR}")
print(f"📁 FRONTEND_DIR : {FRONTEND_DIR}")


# ============================================================
# HELPERS
# ============================================================

def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_load(path: Path, name: str):
    try:
        obj = load_pkl(path)
        print(f"✅ {name} loaded -> {path}")
        return obj
    except Exception as e:
        print(f"⚠️  {name} not found -> {path}")
        print(f"    Reason: {e}")
        return None


# ============================================================
# LOAD HEART FILES
# ============================================================

heart_model_path    = MODELS_DIR / "heart" / "best_model.pkl"
heart_features_path = MODELS_DIR / "heart" / "feature_names.pkl"
heart_meta_path     = MODELS_DIR / "heart" / "meta.pkl"

heart_model    = safe_load(heart_model_path,    "Heart model")
heart_features = safe_load(heart_features_path, "Heart features")
heart_meta     = safe_load(heart_meta_path,     "Heart meta")

heart_threshold = 0.35
if heart_meta and isinstance(heart_meta, dict) and "threshold" in heart_meta:
    heart_threshold = heart_meta["threshold"]

if heart_features:
    print(f"🫀 Heart threshold: {heart_threshold}")
    print(f"🫀 Heart features : {heart_features}")


# ============================================================
# LOAD STROKE FILES
# ============================================================

stroke_model_path    = MODELS_DIR / "stroke" / "best_model.pkl"
stroke_features_path = MODELS_DIR / "stroke" / "feature_list.pkl"
stroke_scaler_path   = MODELS_DIR / "stroke" / "scaler.pkl"

stroke_model    = safe_load(stroke_model_path,    "Stroke model")
stroke_features = safe_load(stroke_features_path, "Stroke feature list")
stroke_scaler   = safe_load(stroke_scaler_path,   "Stroke scaler")
stroke_threshold = 0.5

if stroke_features:
    print(f"🧠 Stroke features: {stroke_features}")


# ============================================================
# LOAD DIABETES FILES
# ============================================================

diabetes_model_path    = MODELS_DIR / "diabetes" / "best_model.pkl"
diabetes_features_path = MODELS_DIR / "diabetes" / "feature_list.pkl"
diabetes_scaler_path   = MODELS_DIR / "diabetes" / "scaler.pkl"

diabetes_model    = safe_load(diabetes_model_path,    "Diabetes model")
diabetes_features = safe_load(diabetes_features_path, "Diabetes feature list")
diabetes_scaler   = safe_load(diabetes_scaler_path,   "Diabetes scaler")
diabetes_threshold = 0.5

if diabetes_features:
    print(f"🩺 Diabetes features: {diabetes_features}")


# ============================================================
# HEART FEATURE ENGINEERING
# ============================================================

def engineer_heart_features(raw: dict) -> dict:
    """
    Frontend sends:
      age, gender, ap_hi, ap_lo, cholesterol, gluc,
      smoke, alco, active, bmi

    We generate:
      pulse_pressure, bp_severity, bmi_age, smoke_age
    """
    raw = dict(raw)

    ap_hi = float(raw["ap_hi"])
    ap_lo = float(raw["ap_lo"])
    age   = float(raw["age"])
    bmi   = float(raw["bmi"])
    smoke = float(raw["smoke"])

    raw["pulse_pressure"] = ap_hi - ap_lo

    if ap_hi >= 160:
        raw["bp_severity"] = 4
    elif ap_hi >= 140:
        raw["bp_severity"] = 3
    elif ap_hi >= 130:
        raw["bp_severity"] = 2
    elif ap_hi >= 120:
        raw["bp_severity"] = 1
    else:
        raw["bp_severity"] = 0

    raw["bmi_age"]   = round(bmi * age, 2)
    raw["smoke_age"] = round(smoke * age, 2)

    return raw


# ============================================================
# GENERIC PREDICTION HELPERS
# ============================================================

def predict_with_dataframe(model, feature_names, raw_data, threshold=0.5, engineer_fn=None):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = dict(raw_data)

    if engineer_fn is not None:
        data = engineer_fn(data)

    if feature_names is not None:
        try:
            df = pd.DataFrame([data])[feature_names]
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    else:
        df = pd.DataFrame([data])

    proba = float(model.predict_proba(df)[0][1])
    pred  = int(proba >= threshold)

    return {
        "prediction": pred,
        "probability": round(proba, 4),
        "risk_level": "HIGH" if pred == 1 else "LOW",
        "threshold": threshold
    }


def predict_with_scaler(model, feature_names, scaler, raw_data, threshold=0.5):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Feature list not loaded")

    data = dict(raw_data)

    try:
        df = pd.DataFrame([data])[feature_names]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    X_scaled = scaler.transform(df)
    proba = float(model.predict_proba(X_scaled)[0][1])
    pred  = int(proba >= threshold)

    return {
        "prediction": pred,
        "probability": round(proba, 4),
        "risk_level": "HIGH" if pred == 1 else "LOW",
        "threshold": threshold
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    try:
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Frontend not found</h1><p>Expected: frontend/index.html</p>"


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "paths": {
            "project_dir": str(PROJECT_DIR),
            "models_dir": str(MODELS_DIR),
            "frontend_dir": str(FRONTEND_DIR),
        },
        "models": {
            "heart_model": "loaded" if heart_model else "not found",
            "heart_features": "loaded" if heart_features else "not found",
            "heart_meta": "loaded" if heart_meta else "not found",

            "stroke_model": "loaded" if stroke_model else "not found",
            "stroke_features": "loaded" if stroke_features else "not found",
            "stroke_scaler": "loaded" if stroke_scaler else "not found",

            "diabetes_model": "loaded" if diabetes_model else "not found",
            "diabetes_features": "loaded" if diabetes_features else "not found",
            "diabetes_scaler": "loaded" if diabetes_scaler else "not found",
        },
        "thresholds": {
            "heart": heart_threshold,
            "stroke": stroke_threshold,
            "diabetes": diabetes_threshold,
        }
    }


# ============================================================
# HEART ENDPOINT
# ============================================================

@app.post("/predict/heart")
def predict_heart(data: dict):
    """
    Expected from frontend:
    {
      age, gender, ap_hi, ap_lo, cholesterol,
      gluc, smoke, alco, active, bmi
    }
    """
    return predict_with_dataframe(
        model=heart_model,
        feature_names=heart_features,
        raw_data=data,
        threshold=heart_threshold,
        engineer_fn=engineer_heart_features
    )


# ============================================================
# STROKE ENDPOINT
# ============================================================

@app.post("/predict/stroke")
def predict_stroke(data: dict):
    """
    Expected keys should match stroke feature_list.pkl
    """
    return predict_with_scaler(
        model=stroke_model,
        feature_names=stroke_features,
        scaler=stroke_scaler,
        raw_data=data,
        threshold=stroke_threshold
    )


# ============================================================
# DIABETES ENDPOINT
# ============================================================

@app.post("/predict/diabetes")
def predict_diabetes(data: dict):
    """
    Expected keys should match diabetes feature_list.pkl
    """
    return predict_with_scaler(
        model=diabetes_model,
        feature_names=diabetes_features,
        scaler=diabetes_scaler,
        raw_data=data,
        threshold=diabetes_threshold
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)