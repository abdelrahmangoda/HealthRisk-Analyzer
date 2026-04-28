from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
import os

app = FastAPI(title="Health Risk Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Load Models --
heart_model = pickle.load(open("models/heart/best_model.pkl", "rb"))
stroke_model = pickle.load(open("models/stroke/best_model.pkl", "rb"))
diabetes_model = pickle.load(open("models/diabetes/best_model.pkl", "rb"))

# -- Serve Frontend --
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()

# -- HEART --
@app.post("/predict/heart")
def predict_heart(data: dict):
    df = pd.DataFrame([data])
    pred = heart_model.predict(df)[0]
    prob = heart_model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(prob)}

# -- STROKE --
@app.post("/predict/stroke")
def predict_stroke(data: dict):
    df = pd.DataFrame([data])
    pred = stroke_model.predict(df)[0]
    prob = stroke_model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(prob)}

# -- DIABETES --
@app.post("/predict/diabetes")
def predict_diabetes(data: dict):
    df = pd.DataFrame([data])
    pred = diabetes_model.predict(df)[0]
    prob = diabetes_model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(prob)}