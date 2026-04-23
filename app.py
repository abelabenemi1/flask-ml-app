from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

app = Flask(__name__)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "refinery_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "oil_data.csv")

# ================= LOAD MODEL (SAFE) =================
if os.path.exists(MODEL_PATH):
    saved_system = joblib.load(MODEL_PATH)
    model = saved_system["model"]
    scaler = saved_system["scaler"]
else:
    model = None
    scaler = None
    print("⚠️ WARNING: Model file not found. Please train first.")

# ================= QUALITY CONTROL =================
SAFE_RANGES = {
    'Density': (0.80, 0.91),
    'API': (23.0, 45.0),
    'Viscosity': (2.7, 15.0)
}

def check_quality(new_row):
    flags = []
    for col, (min_val, max_val) in SAFE_RANGES.items():
        val = new_row.get(col)
        if val is None:
            flags.append(f"MISSING: {col}")
        elif not (min_val <= val <= max_val):
            flags.append(f"OUTLIER: {col} {val} outside {min_val}-{max_val}")
    return flags

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded. Please retrain first."})

        data = request.get_json()

        d = float(data["Density"])
        a = float(data["API"])
        v = float(data["Viscosity"])

        flags = check_quality({
            "Density": d,
            "API": a,
            "Viscosity": v
        })

        log_v = np.log(v) if v > 0 else 0

        X = pd.DataFrame([[d, a, log_v]],
                         columns=["Density", "API", "Log_Viscosity"])

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        return jsonify({
            "Light_Ends": round(float(pred[0]), 2),
            "Mid_Range": round(float(pred[1]), 2),
            "Heavy_Ends": round(float(pred[2]), 2),
            "Total": round(float(sum(pred)), 2),
            "flags": flags
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)})

# ================= ADD DATA =================
@app.route("/add_data", methods=["POST"])
def add_data():
    try:
        data = request.get_json()

        d = float(data["Density"])
        a = float(data["API"])
        v = float(data["Viscosity"])
        light = float(data["Light_Ends"])
        mid = float(data["Mid_Range"])
        heavy = float(data["Heavy_Ends"])

        log_v = np.log(v) if v > 0 else 0

        new_row = {
            "Density": d,
            "API": a,
            "Viscosity": v,
            "Log_Viscosity": log_v,
            "Light_Ends": light,
            "Mid_Range": mid,
            "Heavy_Ends": heavy
        }

        flags = check_quality(new_row)

        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            df = pd.DataFrame(columns=new_row.keys())

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)

        return jsonify({
            "message": "Data saved successfully",
            "flags": flags
        })

    except Exception as e:
        print("Add data error:", e)
        return jsonify({"error": str(e)})

# ================= RETRAIN =================
def retrain_system():
    df = pd.read_csv(DATA_PATH)

    df["Log_Viscosity"] = np.log(df["Viscosity"])

    X = df[["Density", "API", "Log_Viscosity"]]
    y = df[["Light_Ends", "Mid_Range", "Heavy_Ends"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return model, scaler

    joblib.dump({
        "model": model,
        "scaler": scaler
    }, MODEL_PATH)

    return model, scaler, scores

# ================= RETRAIN ROUTE =================
@app.route("/retrain", methods=["POST"])
def retrain():
    global model, scaler
    try:
        model, scaler = retrain_system()
        return jsonify({
            "message": "Model retrained successfully"
        })
    except Exception as e:
        print("Retrain error:", e)
        return jsonify({"error": str(e)})

# ================= START SERVER =================
if __name__ == "__main__":
    app.run()