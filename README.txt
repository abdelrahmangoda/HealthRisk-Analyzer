╔══════════════════════════════════════════════════════════════════╗
║          HEALTH RISK PREDICTOR – SETUP GUIDE                    ║
║          Diabetes · Heart Disease · Stroke                      ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLDER STRUCTURE (place your .pkl files exactly here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

health_predictor/
│
├── app.py                      ← Flask server
├── requirements.txt            ← Python dependencies
├── README.txt                  ← This file
│
├── models/
│   ├── diabetes/
│   │   ├── diabetes_model_final.pkl
│   │   ├── scaler.pkl
│   │   ├── gender_encoder.pkl
│   │   └── smoking_ohe.pkl
│   │
│   ├── heart/
│   │   ├── cardio_best_pipeline__2_.pkl
│   │   ├── feature_names.pkl
│   │   └── meta.pkl
│   │
│   └── stroke/
│       ├── final_stroke_model.pkl
│       └── feature_order.pkl
│
└── templates/
    └── index.html              ← Web interface


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP-BY-STEP: RUN ON YOUR LAPTOP (Windows CMD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Open Command Prompt
  Press  Windows + R  →  type  cmd  →  press Enter

STEP 2 — Navigate to the project folder
  cd path\to\health_predictor

  Example:
  cd C:\Users\YourName\Desktop\health_predictor

STEP 3 — (Recommended) Create a virtual environment
  python -m venv venv
  venv\Scripts\activate

  You will see (venv) at the start of your prompt.

STEP 4 — Install dependencies
  pip install -r requirements.txt

  This installs Flask, scikit-learn, XGBoost, LightGBM, etc.
  Takes 1–3 minutes on first run.

STEP 5 — Start the server
  python app.py

  You should see:
    Starting Health Predictor at http://localhost:5000

STEP 6 — Open Chrome and go to:
  http://localhost:5000

  The web app will load automatically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TO STOP THE SERVER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Press  Ctrl + C  in the CMD window.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• "python is not recognized"
  → Make sure Python is installed and added to PATH.
  → Download from https://python.org  (check "Add to PATH" during install)

• "ModuleNotFoundError: No module named 'xgboost'"
  → Run:  pip install xgboost lightgbm

• "Address already in use" (port 5000 busy)
  → In app.py, change port=5000 to port=5001
  → Then visit http://localhost:5001

• "FileNotFoundError: ... .pkl"
  → Check that all .pkl files are in the correct sub-folders
    (see folder structure above)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This application is for educational and research purposes only.
It is NOT a substitute for professional medical advice,
diagnosis, or treatment.
