"""
Health Risk Predictor – Flask Backend
Serves three ML models: Diabetes, Heart Disease, Stroke
"""
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def _load(path):
    return joblib.load(os.path.join(BASE, path))

# ── Diabetes ──────────────────────────────────────────────────────────────────
# The saved model is XGBoost trained on RAW (unscaled) X_train.
# The scaler.pkl only served LogisticRegression — do NOT use it here.
diabetes_model   = _load('models/diabetes/diabetes_model_final.pkl')
DIABETES_SMOKING = ['No Info', 'current', 'former', 'never', 'not current']

# ── Heart ─────────────────────────────────────────────────────────────────────
heart_pipeline = _load('models/heart/cardio_best_pipeline__2_.pkl')
heart_meta     = _load('models/heart/meta.pkl')
heart_features = heart_meta['feature_names']

# ── Stroke ────────────────────────────────────────────────────────────────────
stroke_model    = _load('models/stroke/final_stroke_model.pkl')
stroke_scaler   = _load('models/stroke/scaler.pkl')
stroke_features = _load('models/stroke/feature_order.pkl')

print("All models loaded.")

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_diabetes(d):
    # Feature order from notebook cell 26:
    # 0:gender  1:age  2:hypertension  3:heart_disease  4:bmi
    # 5:hbA1c_level  6:blood_glucose_level
    # 7:smoking_history_No Info  8:smoking_history_current
    # 9:smoking_history_former  10:smoking_history_never
    # 11:smoking_history_not current
    # XGBoost was trained on unscaled data — return raw array
    gender_enc = 1 if d['gender'] == 'Male' else 0
    smoking    = d['smoking_history']
    smoke_ohe  = [1 if smoking == cat else 0 for cat in DIABETES_SMOKING]
    row = np.array([[
        gender_enc, float(d['age']), int(d['hypertension']),
        int(d['heart_disease']), float(d['bmi']),
        float(d['hbA1c_level']), float(d['blood_glucose_level']),
        *smoke_ohe
    ]])
    return row   # shape (1, 12) — no scaling needed


def preprocess_heart(d):
    age   = float(d['age'])
    bmi   = float(d['bmi'])
    smoke = int(d['smoke'])
    ap_hi = float(d['ap_hi'])
    ap_lo = float(d['ap_lo'])

    pulse_pressure = ap_hi - ap_lo
    bp_severity    = (1 if ap_hi < 120 and ap_lo < 80
                      else 2 if ap_hi < 130 and ap_lo < 80
                      else 3 if ap_hi < 140 or ap_lo < 90
                      else 4)
    row = pd.DataFrame([{
        'age': age, 'gender': int(d['gender']),
        'ap_hi': ap_hi, 'ap_lo': ap_lo,
        'cholesterol': int(d['cholesterol']), 'gluc': int(d['gluc']),
        'smoke': smoke, 'alco': int(d['alco']), 'active': int(d['active']),
        'bmi': bmi, 'pulse_pressure': pulse_pressure,
        'bp_severity': bp_severity, 'bmi_age': bmi * age, 'smoke_age': smoke * age
    }])
    return row.reindex(columns=heart_features, fill_value=0)


def preprocess_stroke(d):
    age     = float(d['age'])
    smoking = d['smoking_status']
    work    = d['work_type']
    row = {
        'gender':                         1 if d['gender'] == 'Male' else 0,
        'age':                            age,
        'hypertension':                   int(d['hypertension']),
        'heart_disease':                  int(d['heart_disease']),
        'ever_married':                   1 if d['ever_married'] == 'Yes' else 0,
        'Residence_type':                 1 if d['Residence_type'] == 'Urban' else 0,
        'avg_glucose_level':              float(d['avg_glucose_level']),
        'bmi':                            float(d['bmi']),
        'work_type_Private':              1 if work == 'Private' else 0,
        'work_type_Self-employed':        1 if work == 'Self-employed' else 0,
        'work_type_children':             1 if work == 'children' else 0,
        'smoking_status_formerly smoked': 1 if smoking == 'formerly smoked' else 0,
        'smoking_status_never smoked':    1 if smoking == 'never smoked' else 0,
        'smoking_status_smokes':          1 if smoking == 'smokes' else 0,
        'is_elderly':                     1 if age >= 65 else 0,
    }
    df = pd.DataFrame([row]).reindex(columns=stroke_features, fill_value=0)
    return stroke_scaler.transform(df.values)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        X    = preprocess_diabetes(request.get_json())
        pred = int(diabetes_model.predict(X)[0])
        prob = float(diabetes_model.predict_proba(X)[0][1])
        return jsonify({'prediction': pred, 'probability': round(prob * 100, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        df        = preprocess_heart(request.get_json())
        threshold = float(heart_meta.get('threshold', 0.5))
        prob      = float(heart_pipeline.predict_proba(df)[0][1])
        pred      = 1 if prob >= threshold else 0
        return jsonify({'prediction': pred, 'probability': round(prob * 100, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/stroke', methods=['POST'])
def predict_stroke():
    try:
        X    = preprocess_stroke(request.get_json())
        pred = int(stroke_model.predict(X)[0])
        prob = float(stroke_model.predict_proba(X)[0][1])
        return jsonify({'prediction': pred, 'probability': round(prob * 100, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n  Starting Health Predictor at http://localhost:5000\n")
    app.run(debug=True, port=5000)
