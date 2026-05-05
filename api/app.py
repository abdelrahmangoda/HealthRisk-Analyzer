# ============================================================
# app.py — Health Risk Analyzer API  (v2 — improved)
# ============================================================

import os
import pickle
import joblib
import numpy as np
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
    """Try pickle first, then joblib."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    try:
        return joblib.load(path)
    except Exception:
        pass
    raise ValueError(f"Could not load: {path}")


def safe_load(path: Path, name: str):
    try:
        obj = load_any(path)
        print(f"✅ {name} loaded   → {path.name}")
        return obj
    except Exception as e:
        print(f"⚠️  {name} FAILED  → {path.name} | {e}")
        return None


def is_real_scaler(obj) -> bool:
    return (
        obj is not None
        and hasattr(obj, "transform")
        and callable(obj.transform)
        and not isinstance(obj, np.ndarray)
    )


def type_label(obj) -> str:
    return getattr(type(obj), "__name__", "unknown")


# ============================================================
# LOAD MODELS
# ============================================================

heart_model     = safe_load(MODELS_DIR / "heart"    / "best_model.pkl",   "Heart model")
heart_features  = safe_load(MODELS_DIR / "heart"    / "feature_names.pkl","Heart features")
heart_meta      = safe_load(MODELS_DIR / "heart"    / "meta.pkl",         "Heart meta")

stroke_model    = safe_load(MODELS_DIR / "stroke"   / "best_model.pkl",   "Stroke model")
stroke_features = safe_load(MODELS_DIR / "stroke"   / "feature_list.pkl", "Stroke features")
stroke_scaler   = safe_load(MODELS_DIR / "stroke"   / "scaler.pkl",       "Stroke scaler")

diabetes_model    = safe_load(MODELS_DIR / "diabetes" / "best_model.pkl",   "Diabetes model")
diabetes_features = safe_load(MODELS_DIR / "diabetes" / "feature_list.pkl", "Diabetes features")
diabetes_scaler   = safe_load(MODELS_DIR / "diabetes" / "scaler.pkl",       "Diabetes scaler")

# ── thresholds ──────────────────────────────────────────────
heart_threshold = 0.35
if isinstance(heart_meta, dict) and "threshold" in heart_meta:
    heart_threshold = heart_meta["threshold"]

stroke_threshold   = 0.5
diabetes_threshold = 0.5

print(f"\n🫀 Heart    threshold={heart_threshold}  features={heart_features}")
print(f"🧠 Stroke   threshold={stroke_threshold}  scaler={'real' if is_real_scaler(stroke_scaler) else type_label(stroke_scaler)}")
print(f"🩺 Diabetes threshold={diabetes_threshold} scaler={'real' if is_real_scaler(diabetes_scaler) else type_label(diabetes_scaler)}\n")


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_heart(data: dict):
    """
    فحص منطقية بيانات القلب قبل الإرسال للموديل.
    يرفع HTTPException(422) عند وجود خطأ.
    """
    age   = float(data.get("age", 0))
    bmi   = float(data.get("bmi", 0))
    ap_hi = float(data.get("ap_hi", 0))
    ap_lo = float(data.get("ap_lo", 0))

    errors = []

    if not (1 <= age <= 120):
        errors.append("age must be 1–120")

    if not (10 <= bmi <= 70):
        errors.append(f"bmi={bmi} out of range 10–70")

    if not (60 <= ap_hi <= 250):
        errors.append(f"ap_hi={ap_hi} out of range 60–250")

    if not (40 <= ap_lo <= 200):
        errors.append(f"ap_lo={ap_lo} out of range 40–200")

    if ap_hi <= ap_lo:
        errors.append("ap_hi must be greater than ap_lo")

    if (ap_hi - ap_lo) < 20:
        errors.append(f"pulse_pressure={ap_hi-ap_lo} too low — check readings")

    if errors:
        raise HTTPException(422, {"validation_errors": errors})


def validate_stroke(data: dict):
    age     = float(data.get("age", 0))
    bmi     = float(data.get("bmi", 0))
    glucose = float(data.get("avg_glucose_level", 0))

    errors = []

    if not (1 <= age <= 120):
        errors.append("age must be 1–120")

    if not (10 <= bmi <= 70):
        errors.append(f"bmi={bmi} out of range 10–70")

    if not (50 <= glucose <= 400):
        errors.append(f"avg_glucose_level={glucose} out of range 50–400")

    if errors:
        raise HTTPException(422, {"validation_errors": errors})


def validate_diabetes(data: dict):
    age     = float(data.get("age", 0))
    bmi     = float(data.get("bmi", 0))
    hba1c   = float(data.get("hbA1c_level", 0))
    glucose = float(data.get("blood_glucose_level", 0))

    errors = []

    if not (1 <= age <= 120):
        errors.append("age must be 1–120")

    if not (10 <= bmi <= 70):
        errors.append(f"bmi={bmi} out of range 10–70")

    if not (3 <= hba1c <= 15):
        errors.append(f"hbA1c_level={hba1c} out of range 3–15")

    if not (50 <= glucose <= 400):
        errors.append(f"blood_glucose_level={glucose} out of range 50–400")

    # تحقق من تناسق HbA1c مع الجلوكوز
    if hba1c > 9 and glucose < 100:
        errors.append(
            f"Inconsistent: hbA1c={hba1c}% (high) with glucose={glucose} mg/dL (low)"
        )

    if glucose > 300 and hba1c < 6:
        errors.append(
            f"Inconsistent: glucose={glucose} mg/dL (very high) with hbA1c={hba1c}% (normal)"
        )

    if errors:
        raise HTTPException(422, {"validation_errors": errors})


# ============================================================
# HEART — FEATURE ENGINEERING
# ============================================================
#
# الموديل المحفوظ اتدرب على 14 features بالظبط:
#   age, gender, ap_hi, ap_lo, cholesterol, gluc,
#   smoke, alco, active, bmi,
#   pulse_pressure, bp_severity, bmi_age, smoke_age
#
# Frontend بيبعت 10 raw fields.
# Backend بيضيف 4 derived features.
# الـ feature_names.pkl بيرتب الكل بالترتيب الصح.
# ============================================================

def engineer_heart_features(raw: dict) -> dict:
    """
    Adds 4 derived features to match the saved model exactly.
    Total: 14 features (10 raw + 4 derived).
    feature_names.pkl controls the final column order.
    """
    raw = dict(raw)

    ap_hi = float(raw["ap_hi"])
    ap_lo = float(raw["ap_lo"])
    age   = float(raw["age"])
    bmi   = float(raw["bmi"])
    smoke = float(raw.get("smoke", 0))

    # pulse pressure — فرق الضغط
    raw["pulse_pressure"] = ap_hi - ap_lo

    # bp_severity — تصنيف نوعي لشدة الضغط الانقباضي
    if   ap_hi >= 160: raw["bp_severity"] = 4
    elif ap_hi >= 140: raw["bp_severity"] = 3
    elif ap_hi >= 130: raw["bp_severity"] = 2
    elif ap_hi >= 120: raw["bp_severity"] = 1
    else:              raw["bp_severity"] = 0

    # bmi_age — تفاعل الوزن مع العمر
    raw["bmi_age"]   = round(bmi * age, 2)

    # smoke_age — تراكم التدخين مع العمر
    raw["smoke_age"] = round(smoke * age, 2)

    return raw


# ============================================================
# DIABETES — SMOKING NORMALIZATION
# ============================================================

DIABETES_SMOKE_FIELDS = [
    "smoking_history_No Info",
    "smoking_history_current",
    "smoking_history_ever",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_not current",
]

def normalize_diabetes_smoking(data: dict) -> dict:
    """
    Ensures all 6 one-hot smoking fields are present and mutually exclusive.
    """
    data = dict(data)
    total = sum(int(data.get(f, 0)) for f in DIABETES_SMOKE_FIELDS)

    if total > 1:
        found = False
        for f in DIABETES_SMOKE_FIELDS:
            if int(data.get(f, 0)) == 1:
                if found:
                    data[f] = 0
                else:
                    found = True

    # إذا لم يُحدَّد أي خيار → No Info
    if total == 0:
        data["smoking_history_No Info"] = 1

    for f in DIABETES_SMOKE_FIELDS:
        data.setdefault(f, 0)

    return data


# ============================================================
# CORE PREDICTION HELPERS
# ============================================================

def _build_df(data: dict, feature_names) -> pd.DataFrame:
    if feature_names is not None:
        missing = [f for f in feature_names if f not in data]
        if missing:
            raise HTTPException(422, f"Missing features: {missing}")
        return pd.DataFrame([data])[feature_names]
    return pd.DataFrame([data])


def _apply_scaler(df: pd.DataFrame, scaler):
    if is_real_scaler(scaler):
        try:
            return scaler.transform(df)
        except Exception as e:
            print(f"   ⚠️  scaler.transform failed: {e} — using raw")
    elif scaler is not None:
        print(f"   ⚠️  scaler is {type_label(scaler)}, not a real scaler — using raw")
    return df.values


def _predict_proba(model, X, df: pd.DataFrame) -> float:
    try:
        return float(model.predict_proba(X)[0][1])
    except Exception:
        return float(model.predict_proba(df)[0][1])


def _result(proba: float, threshold: float) -> dict:
    pred = int(proba >= threshold)
    return {
        "prediction" : pred,
        "probability": round(proba, 4),
        "risk_level" : "HIGH" if pred == 1 else "LOW",
        "threshold"  : threshold,
    }


# ============================================================
# PUBLIC PREDICTION FUNCTIONS
# ============================================================

def predict_direct(model, feature_names, raw_data: dict,
                   threshold=0.5, engineer_fn=None) -> dict:
    if model is None:
        raise HTTPException(503, "Model not loaded")

    data = dict(raw_data)
    if engineer_fn:
        data = engineer_fn(data)

    df    = _build_df(data, feature_names)
    proba = _predict_proba(model, df, df)
    return _result(proba, threshold)


def predict_scaled(model, feature_names, scaler, raw_data: dict,
                   threshold=0.5, normalize_fn=None) -> dict:
    if model is None:
        raise HTTPException(503, "Model not loaded")

    data = dict(raw_data)
    if normalize_fn:
        data = normalize_fn(data)

    df = _build_df(data, feature_names)
    X  = _apply_scaler(df, scaler)
    proba = _predict_proba(model, X, df)
    return _result(proba, threshold)


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
        return (
            "<h1>Frontend not found</h1>"
            "<p>Expected: frontend/index.html</p>"
            "<p><a href='/health'>/health</a> | <a href='/docs'>/docs</a></p>"
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    def scaler_status(s):
        if is_real_scaler(s): return "✅ real scaler"
        if s is not None:      return f"⚠️ {type_label(s)}"
        return "❌ not loaded"

    return {
        "status": "running",
        "paths": {
            "project" : str(PROJECT_DIR),
            "models"  : str(MODELS_DIR),
            "frontend": str(FRONTEND_DIR),
        },
        "models": {
            "heart": {
                "model"    : "✅" if heart_model    else "❌",
                "features" : "✅" if heart_features  else "❌",
                "meta"     : "✅" if heart_meta      else "❌",
                "threshold": heart_threshold,
            },
            "stroke": {
                "model"    : "✅" if stroke_model    else "❌",
                "features" : "✅" if stroke_features  else "❌",
                "scaler"   : scaler_status(stroke_scaler),
                "threshold": stroke_threshold,
            },
            "diabetes": {
                "model"    : "✅" if diabetes_model    else "❌",
                "features" : "✅" if diabetes_features  else "❌",
                "scaler"   : scaler_status(diabetes_scaler),
                "threshold": diabetes_threshold,
            },
        },
    }


# ============================================================
# ENDPOINTS
# ============================================================

def apply_heart_adjustments(base_proba: float, raw: dict) -> float:
    """
    تعديل يدوي على نتيجة الموديل لتعويض ضعف تأثير التدخين والجلوكوز.

    منطق التعديل:
    - الـ boost مبني على الـ base_proba نفسها — كلما ارتفع الخطر الأصلي
      كان التعديل أكبر بالمطلق (لكن ثابت نسبياً)
    - الحد الأقصى للنتيجة النهائية 0.97 منعاً للتشبع

    التدخين:
      smoke=1 → +2% إذا base < 0.4 | +3% إذا base >= 0.4 | +5% إذا base >= 0.7
    الكحول:
      alco=1  → +1.5% إذا base < 0.4 | +2% إذا base >= 0.4 | +3% إذا base >= 0.7
    الجلوكوز:
      gluc=2 (مرتفع)     → +2%
      gluc=3 (مرتفع جداً) → +4%
    """
    proba  = base_proba
    smoke  = float(raw.get("smoke", 0))
    alco   = float(raw.get("alco",  0))
    gluc   = float(raw.get("gluc",  1))

    # ── تدخين ──
    if smoke == 1:
        if   proba >= 0.70: proba += 0.050
        elif proba >= 0.40: proba += 0.030
        else:               proba += 0.020

    # ── كحول ──
    if alco == 1:
        if   proba >= 0.70: proba += 0.030
        elif proba >= 0.40: proba += 0.020
        else:               proba += 0.015

    # ── جلوكوز ──
    if   gluc == 3: proba += 0.040
    elif gluc == 2: proba += 0.020

    return min(proba, 0.97)


@app.post("/predict/heart")
def predict_heart(data: dict):
    """
    Accepts 10 raw fields. Backend adds 4 engineered features.
    Applies manual adjustments for smoking, alcohol, glucose.
    """
    validate_heart(data)
    result = predict_direct(
        model         = heart_model,
        feature_names = heart_features,
        raw_data      = data,
        threshold     = heart_threshold,
        engineer_fn   = engineer_heart_features,
    )

    # تطبيق التعديل اليدوي
    adjusted = apply_heart_adjustments(result["probability"], data)
    result["probability"]      = round(adjusted, 4)
    result["prediction"]       = int(adjusted >= heart_threshold)
    result["risk_level"]       = "HIGH" if result["prediction"] == 1 else "LOW"
    result["probability_raw"]  = round(result["probability"], 4)   # للـ debug

    return result


@app.post("/predict/stroke")
def predict_stroke(data: dict):
    """Accepts 15 fields. Validates before prediction."""
    validate_stroke(data)
    return predict_scaled(
        model         = stroke_model,
        feature_names = stroke_features,
        scaler        = stroke_scaler,
        raw_data      = data,
        threshold     = stroke_threshold,
    )


@app.post("/predict/diabetes")
def predict_diabetes(data: dict):
    """
    Accepts 13 fields. Smoking mapped as 6 one-hot fields.
    Validates consistency of HbA1c vs glucose before prediction.
    """
    validate_diabetes(data)
    return predict_scaled(
        model         = diabetes_model,
        feature_names = diabetes_features,
        scaler        = diabetes_scaler,
        raw_data      = data,
        threshold     = diabetes_threshold,
        normalize_fn  = normalize_diabetes_smoking,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Health Risk Analyzer API  (v2)")
    print("   Frontend : http://127.0.0.1:8000")
    print("   Health   : http://127.0.0.1:8000/health")
    print("   Docs     : http://127.0.0.1:8000/docs\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)