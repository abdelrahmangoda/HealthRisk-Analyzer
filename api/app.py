# ============================================================
# app.py — Health Risk Analyzer API
# ============================================================

import os
import pickle
import joblib
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

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
MODELS_DIR   = PROJECT_DIR / "models"
FRONTEND_DIR = PROJECT_DIR / "frontend"

print(f"📁 BASE_DIR     : {BASE_DIR}")
print(f"📁 PROJECT_DIR  : {PROJECT_DIR}")
print(f"📁 MODELS_DIR   : {MODELS_DIR}")
print(f"📁 FRONTEND_DIR : {FRONTEND_DIR}")


# ============================================================
# HELPERS
# ============================================================

def load_any(path: Path):
    """Try pickle first, then joblib"""
    # Try pickle
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    except Exception:
        pass

    # Try joblib
    try:
        obj = joblib.load(path)
        return obj
    except Exception:
        pass

    raise ValueError(f"Could not load with pickle or joblib: {path}")


def safe_load(path: Path, name: str):
    """Load file safely with error handling"""
    try:
        obj = load_any(path)
        print(f"✅ {name} loaded -> {path}")
        return obj
    except Exception as e:
        print(f"⚠️  {name} FAILED -> {path}")
        print(f"    Reason: {e}")
        return None


# ============================================================
# LOAD HEART
# ============================================================

heart_model    = safe_load(MODELS_DIR / "heart" / "best_model.pkl",      "Heart model")
heart_features = safe_load(MODELS_DIR / "heart" / "feature_names.pkl",   "Heart features")
heart_meta     = safe_load(MODELS_DIR / "heart" / "meta.pkl",            "Heart meta")

heart_threshold = 0.35
if heart_meta and isinstance(heart_meta, dict) and "threshold" in heart_meta:
    heart_threshold = heart_meta["threshold"]

if heart_features:
    print(f"🫀 Heart threshold : {heart_threshold}")
    print(f"🫀 Heart features  : {heart_features}")


# ============================================================
# LOAD STROKE
# ============================================================

stroke_model    = safe_load(MODELS_DIR / "stroke" / "best_model.pkl",      "Stroke model")
stroke_features = safe_load(MODELS_DIR / "stroke" / "feature_list.pkl",    "Stroke features")
stroke_scaler   = safe_load(MODELS_DIR / "stroke" / "scaler.pkl",          "Stroke scaler")
stroke_threshold = 0.5

if stroke_features:
    print(f"🧠 Stroke features : {stroke_features}")


# ============================================================
# LOAD DIABETES
# ============================================================

diabetes_model    = safe_load(MODELS_DIR / "diabetes" / "best_model.pkl",      "Diabetes model")
diabetes_features = safe_load(MODELS_DIR / "diabetes" / "feature_list.pkl",    "Diabetes features")
diabetes_scaler   = safe_load(MODELS_DIR / "diabetes" / "scaler.pkl",          "Diabetes scaler")
diabetes_threshold = 0.5

if diabetes_features:
    print(f"🩺 Diabetes features : {diabetes_features}")


# ============================================================
# HEART FEATURE ENGINEERING
# ============================================================

def engineer_heart_features(raw: dict) -> dict:
    """
    Frontend sends 10 fields:
      age, gender, ap_hi, ap_lo, cholesterol,
      gluc, smoke, alco, active, bmi

    We calculate 4 more:
      pulse_pressure, bp_severity, bmi_age, smoke_age
    """
    raw = dict(raw)

    ap_hi = float(raw["ap_hi"])
    ap_lo = float(raw["ap_lo"])
    age   = float(raw["age"])
    bmi   = float(raw["bmi"])
    smoke = float(raw["smoke"])

    # Pulse Pressure
    raw["pulse_pressure"] = ap_hi - ap_lo

    # BP Severity (5 levels)
    if   ap_hi >= 160: raw["bp_severity"] = 4
    elif ap_hi >= 140: raw["bp_severity"] = 3
    elif ap_hi >= 130: raw["bp_severity"] = 2
    elif ap_hi >= 120: raw["bp_severity"] = 1
    else:              raw["bp_severity"] = 0

    # Interactions
    raw["bmi_age"]   = round(bmi * age, 2)
    raw["smoke_age"] = round(smoke * age, 2)

    return raw


# ============================================================
# PREDICTION — WITHOUT SCALER (Heart)
# ============================================================

def predict_direct(model, feature_names, raw_data,
                   threshold=0.5, engineer_fn=None):
    """
    Predict directly (no scaler needed).
    Used for Heart model.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    data = dict(raw_data)

    if engineer_fn is not None:
        data = engineer_fn(data)

    if feature_names is not None:
        try:
            df = pd.DataFrame([data])[feature_names]
        except KeyError as e:
            raise HTTPException(422, f"Missing feature: {e}")
    else:
        df = pd.DataFrame([data])

    proba = float(model.predict_proba(df)[0][1])
    pred  = int(proba >= threshold)

    return {
        "prediction" : pred,
        "probability": round(proba, 4),
        "risk_level" : "HIGH" if pred == 1 else "LOW",
        "threshold"  : threshold,
    }


# ============================================================
# PREDICTION — WITH SCALER (Stroke, Diabetes)
# ============================================================

def predict_scaled(model, feature_names, scaler, raw_data,
                   threshold=0.5):
    """
    Predict with scaler.transform() first.
    Used for Stroke and Diabetes models.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    data = dict(raw_data)

    if feature_names is not None:
        try:
            df = pd.DataFrame([data])[feature_names]
        except KeyError as e:
            raise HTTPException(422, f"Missing feature: {e}")
    else:
        df = pd.DataFrame([data])

    # Scale if scaler exists
    if scaler is not None:
        try:
            X = scaler.transform(df)
        except Exception as e:
            raise HTTPException(
                500, f"Scaler error: {e}. Trying without scaler."
            )
    else:
        # No scaler → use raw
        X = df

    proba = float(model.predict_proba(X)[0][1])
    pred  = int(proba >= threshold)

    return {
        "prediction" : pred,
        "probability": round(proba, 4),
        "risk_level" : "HIGH" if pred == 1 else "LOW",
        "threshold"  : threshold,
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
        return """
        <h1>Frontend not found</h1>
        <p>Expected: frontend/index.html</p>
        <p>Go to <a href='/health'>/health</a> to check API status</p>
        """


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "paths": {
            "project_dir" : str(PROJECT_DIR),
            "models_dir"  : str(MODELS_DIR),
            "frontend_dir": str(FRONTEND_DIR),
        },
        "models": {
            "heart": {
                "model"   : "✅" if heart_model    else "❌",
                "features": "✅" if heart_features  else "❌",
                "meta"    : "✅" if heart_meta      else "❌",
            },
            "stroke": {
                "model"   : "✅" if stroke_model    else "❌",
                "features": "✅" if stroke_features  else "❌",
                "scaler"  : "✅" if stroke_scaler    else "❌",
            },
            "diabetes": {
                "model"   : "✅" if diabetes_model    else "❌",
                "features": "✅" if diabetes_features  else "❌",
                "scaler"  : "✅" if diabetes_scaler    else "❌",
            },
        },
        "thresholds": {
            "heart"   : heart_threshold,
            "stroke"  : stroke_threshold,
            "diabetes": diabetes_threshold,
        },
    }


# ============================================================
# HEART ENDPOINT
# ============================================================

@app.post("/predict/heart")
def predict_heart(data: dict):
    """
    Input (10 fields from frontend):
    {
      "age": 50, "gender": 1,
      "ap_hi": 140, "ap_lo": 90,
      "cholesterol": 2, "gluc": 1,
      "smoke": 0, "alco": 0,
      "active": 1, "bmi": 28.0
    }

    → engineer_heart_features adds 4 more
    → total 14 features for prediction
    """
    return predict_direct(
        model        = heart_model,
        feature_names= heart_features,
        raw_data     = data,
        threshold    = heart_threshold,
        engineer_fn  = engineer_heart_features,
    )


# ============================================================
# STROKE ENDPOINT
# ============================================================

@app.post("/predict/stroke")
def predict_stroke(data: dict):
    """
    Input (15 fields from frontend):
    {
      "gender": 1, "age": 50,
      "hypertension": 0, "heart_disease": 0,
      "ever_married": 1, "Residence_type": 1,
      "avg_glucose_level": 100, "bmi": 25,
      "work_type_Private": 1,
      "work_type_Self-employed": 0,
      "work_type_children": 0,
      "smoking_status_formerly smoked": 0,
      "smoking_status_never smoked": 1,
      "smoking_status_smokes": 0,
      "is_elderly": 0
    }
    """
    return predict_scaled(
        model        = stroke_model,
        feature_names= stroke_features,
        scaler       = stroke_scaler,
        raw_data     = data,
        threshold    = stroke_threshold,
    )


# ============================================================
# DIABETES ENDPOINT
# ============================================================

@app.post("/predict/diabetes")
def predict_diabetes(data: dict):
    """
    Input (13 fields from frontend):
    {
      "gender": 1, "age": 40,
      "hypertension": 0, "heart_disease": 0,
      "bmi": 27, "hbA1c_level": 5.5,
      "blood_glucose_level": 100,
      "smoking_history_No Info": 0,
      "smoking_history_current": 0,
      "smoking_history_ever": 0,
      "smoking_history_former": 0,
      "smoking_history_never": 1,
      "smoking_history_not current": 0
    }
    """
    return predict_scaled(
        model        = diabetes_model,
        feature_names= diabetes_features,
        scaler       = diabetes_scaler,
        raw_data     = data,
        threshold    = diabetes_threshold,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting server...")
    print("   Open: http://127.0.0.1:8000")
    print("   Health: http://127.0.0.1:8000/health")
    print("   Docs: http://127.0.0.1:8000/docs\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)