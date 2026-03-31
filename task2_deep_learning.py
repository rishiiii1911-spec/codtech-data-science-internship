# =============================================================================
# CODTECH INTERNSHIP - TASK 2: DEEP LEARNING PROJECT
# Dataset : Titanic (same as Task 1)
# Tools   : pandas, scikit-learn, TensorFlow/Keras, matplotlib
# Goal    : Build a deep learning model for binary classification (Survived?)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for all environments)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ──────────────────────────────────────────────
# STEP 1: LOAD & PREPROCESS  (reusing Task 1 logic)
# ──────────────────────────────────────────────
def load_and_preprocess(url: str):
    """
    Load Titanic data and apply the same ETL pipeline from Task 1.
    Returns train/test splits ready for the neural network.
    """
    print("📥  Loading and preprocessing data ...")

    df = pd.read_csv(url)

    # Drop low-value columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df = df.drop(columns=["SibSp", "Parch"])

    # Split features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Identify column types
    numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Build preprocessing pipeline
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

    X_processed = preprocessor.fit_transform(X)

    # Train / test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y.values, test_size=0.2, random_state=42, stratify=y
    )

    print(f"    ✅  Train: {X_train.shape}, Test: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# STEP 2: BUILD THE NEURAL NETWORK
# ──────────────────────────────────────────────
def build_model(input_dim: int) -> keras.Model:
    """
    Build a fully-connected (dense) neural network for binary classification.
    Architecture:
        Input  → Dense(64, ReLU) → Dropout(0.3)
               → Dense(32, ReLU) → Dropout(0.2)
               → Dense(1, Sigmoid)   ← outputs survival probability
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Hidden layer 1
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),   # randomly disable 30% of neurons → reduces overfitting

        # Hidden layer 2
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),

        # Output layer — sigmoid squashes output to [0, 1] (survival probability)
        layers.Dense(1, activation="sigmoid")
    ])

    # Compile: binary crossentropy loss for binary classification
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("🧠  Model architecture:")
    model.summary()
    print()
    return model


# ──────────────────────────────────────────────
# STEP 3: TRAIN THE MODEL
# ──────────────────────────────────────────────
def train_model(model, X_train, y_train):
    """
    Train the neural network.
    Uses EarlyStopping to halt training when validation loss stops improving.
    """
    print("🏋️  Training the model ...")

    # EarlyStopping: stop if val_loss doesn't improve for 10 epochs in a row
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,               # maximum epochs
        batch_size=32,
        validation_split=0.15,    # 15% of training data used for validation
        callbacks=[early_stop],
        verbose=1
    )

    print(f"\n    ✅  Training complete — stopped at epoch {len(history.history['loss'])}\n")
    return history


# ──────────────────────────────────────────────
# STEP 4: EVALUATE & VISUALIZE RESULTS
# ──────────────────────────────────────────────
def evaluate_and_visualize(model, history, X_test, y_test):
    """
    Evaluate the model on the test set and produce three visualizations:
      1. Training vs Validation Accuracy curve
      2. Training vs Validation Loss curve
      3. Confusion Matrix
    """
    print("📊  Evaluating model on test set ...")

    # --- 4a. Test set performance -------------------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"    Test Accuracy : {test_acc:.4f}")
    print(f"    Test Loss     : {test_loss:.4f}\n")

    # Predictions (threshold 0.5)
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    print("    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))

    # --- 4b. Plot 1 & 2: Accuracy and Loss curves ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CODTECH Task 2 — Deep Learning Training Results", fontsize=13, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2, color="tomato")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--", color="coral")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("    💾  Saved → training_curves.png")

    # --- 4c. Plot 3: Confusion Matrix ----------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not survive", "Survived"])
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Confusion Matrix — Test Set", fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("    💾  Saved → confusion_matrix.png\n")

    return test_acc


# ──────────────────────────────────────────────
# STEP 5: SAVE THE TRAINED MODEL
# ──────────────────────────────────────────────
def save_model(model, path: str = "titanic_model.keras"):
    """Save the trained model to disk for use in Task 3 (Flask deployment)."""
    model.save(path)
    print(f"✅  Model saved → '{path}'  (use this in Task 3!)\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    DATA_URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

    # Run all steps
    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_URL)
    model   = build_model(input_dim=X_train.shape[1])
    history = train_model(model, X_train, y_train)
    acc     = evaluate_and_visualize(model, history, X_test, y_test)
    save_model(model)

    print("=" * 60)
    print(f"  Task 2 complete!  Final test accuracy: {acc:.2%} 🎉")
    print("  Files generated:")
    print("    • titanic_model.keras   ← saved model for Task 3")
    print("    • training_curves.png   ← accuracy & loss plots")
    print("    • confusion_matrix.png  ← prediction results")
    print("=" * 60)
