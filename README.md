# CODTECH Data Science Internship

All four tasks completed as part of the CODTECH Data Science Internship program.

---

## Task 1 — Data Pipeline Development
**File:** `task1_data_pipeline.py`

Automated ETL pipeline using **pandas** and **scikit-learn** on the Titanic dataset.
- Extract: loads data from a public URL
- Transform: cleans data, engineers features, imputes missing values, scales and encodes
- Load: saves processed data to `processed_titanic.csv`

```bash
pip install pandas scikit-learn
python task1_data_pipeline.py
```

---

## Task 2 — Deep Learning Project
**File:** `task2_deep_learning.py`

Binary classification neural network using **TensorFlow/Keras** to predict Titanic survival.
- Architecture: Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid)
- Outputs: `titanic_model.keras`, `training_curves.png`, `confusion_matrix.png`

```bash
pip install tensorflow pandas scikit-learn matplotlib
python task2_deep_learning.py
```

---

## Task 3 — End-to-End Deployment
**File:** `task3_flask_app.py`

**Flask** REST API that serves the trained Task 2 model with a browser UI.
- `GET  /`        → Web UI to input passenger details and get prediction
- `POST /predict` → JSON API endpoint
- `GET  /health`  → Health check

```bash
pip install flask tensorflow pandas scikit-learn
python task3_flask_app.py
# Open http://127.0.0.1:5000
```

---

## Task 4 — Optimization Model
**File:** `task4_optimization.py`

Production planning optimization using **PuLP** (Linear Programming).
- Maximizes factory profit subject to Wood, Labor, and Machine hour constraints
- Outputs: optimal production quantities + `optimization_results.png`

```bash
pip install pulp pandas matplotlib
python task4_optimization.py
```

---

## Run Order
```
task1 → task2 → task3 → task4 (independent)
```
Tasks 1, 2, and 3 are connected (same dataset and model). Task 4 is standalone.
