# =============================================================================
# CODTECH INTERNSHIP - TASK 3: END-TO-END DATA SCIENCE PROJECT
# Tools   : Flask, TensorFlow, pandas, scikit-learn
# Goal    : Deploy the Task 2 model as a REST API
#
# HOW TO RUN:
#   1. Make sure task2_deep_learning.py has been run first (generates titanic_model.keras)
#   2. pip install flask tensorflow pandas scikit-learn
#   3. python task3_flask_app.py
#   4. Open browser → http://127.0.0.1:5000
#      OR send a POST request to http://127.0.0.1:5000/predict
# =============================================================================

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────
app = Flask(__name__)

# Load the trained model saved by Task 2
MODEL_PATH = "titanic_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅  Model loaded from '{MODEL_PATH}'")

# Rebuild the same preprocessor used in Task 1 & 2
# (fit on the same training data so transformations are identical)
DATA_URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

def build_preprocessor():
    """Rebuild and fit the preprocessing pipeline on the original training data."""
    df = pd.read_csv(DATA_URL)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"], errors="ignore")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df = df.drop(columns=["SibSp", "Parch"])

    numeric_features     = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
        ]), categorical_features)
    ])
    preprocessor.fit(df)
    return preprocessor

preprocessor = build_preprocessor()
print("✅  Preprocessor ready\n")


# ──────────────────────────────────────────────
# HTML PAGE  (simple web UI to test the API)
# ──────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survival Predictor — CODTECH Task 3</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto; background: #f5f5f5; }
        h2   { color: #c0392b; }
        label { display: block; margin-top: 12px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; margin-top: 4px; border-radius: 4px; border: 1px solid #ccc; }
        button { margin-top: 20px; padding: 10px 24px; background: #c0392b; color: white;
                 border: none; border-radius: 4px; cursor: pointer; font-size: 15px; }
        #result { margin-top: 20px; padding: 14px; border-radius: 6px; font-size: 16px; }
        .survived    { background: #d5f5e3; color: #1e8449; }
        .not-survived{ background: #fadbd8; color: #922b21; }
    </style>
</head>
<body>
    <h2>🚢 Titanic Survival Predictor</h2>
    <p>CODTECH Internship — Task 3 (Flask Deployment)</p>

    <label>Passenger Class (Pclass)</label>
    <select id="pclass">
        <option value="1">1st Class</option>
        <option value="2">2nd Class</option>
        <option value="3" selected>3rd Class</option>
    </select>

    <label>Sex</label>
    <select id="sex">
        <option value="male">Male</option>
        <option value="female">Female</option>
    </select>

    <label>Age</label>
    <input type="number" id="age" value="28" min="0" max="100">

    <label>Fare (£)</label>
    <input type="number" id="fare" value="15" min="0">

    <label>Embarked</label>
    <select id="embarked">
        <option value="S" selected>Southampton (S)</option>
        <option value="C">Cherbourg (C)</option>
        <option value="Q">Queenstown (Q)</option>
    </select>

    <label>SibSp (siblings/spouses aboard)</label>
    <input type="number" id="sibsp" value="0" min="0">

    <label>Parch (parents/children aboard)</label>
    <input type="number" id="parch" value="0" min="0">

    <button onclick="predict()">Predict Survival</button>

    <div id="result"></div>

    <script>
        async function predict() {
            const data = {
                Pclass:   parseInt(document.getElementById("pclass").value),
                Sex:      document.getElementById("sex").value,
                Age:      parseFloat(document.getElementById("age").value),
                Fare:     parseFloat(document.getElementById("fare").value),
                Embarked: document.getElementById("embarked").value,
                SibSp:    parseInt(document.getElementById("sibsp").value),
                Parch:    parseInt(document.getElementById("parch").value)
            };

            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const result = await resp.json();
            const div = document.getElementById("result");
            if (result.survived) {
                div.className = "survived";
                div.innerHTML = `✅ <b>Survived</b> — Probability: ${(result.probability * 100).toFixed(1)}%`;
            } else {
                div.className = "not-survived";
                div.innerHTML = `❌ <b>Did not survive</b> — Probability: ${(result.probability * 100).toFixed(1)}%`;
            }
        }
    </script>
</body>
</html>
"""


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the simple web UI."""
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON with passenger details, returns survival prediction.

    Example request body:
    {
        "Pclass": 3, "Sex": "male", "Age": 22,
        "Fare": 7.25, "Embarked": "S", "SibSp": 1, "Parch": 0
    }
    """
    data = request.get_json()

    # Build a single-row DataFrame matching the training schema
    df_input = pd.DataFrame([{
        "Pclass":   data.get("Pclass", 3),
        "Sex":      data.get("Sex", "male"),
        "Age":      data.get("Age", 28),
        "Fare":     data.get("Fare", 15),
        "Embarked": data.get("Embarked", "S"),
        "SibSp":    data.get("SibSp", 0),
        "Parch":    data.get("Parch", 0),
    }])

    # Apply the same preprocessing pipeline
    df_input["FamilySize"] = df_input["SibSp"] + df_input["Parch"] + 1
    df_input = df_input.drop(columns=["SibSp", "Parch"])
    X = preprocessor.transform(df_input)

    # Get prediction
    prob     = float(model.predict(X, verbose=0)[0][0])
    survived = prob >= 0.5

    return jsonify({
        "survived":    bool(survived),
        "probability": round(prob, 4),
        "message":     "Survived! 🎉" if survived else "Did not survive 😢"
    })


@app.route("/health")
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "model": MODEL_PATH})


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  Starting Flask server ...")
    print("    Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
