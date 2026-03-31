# =============================================================================
# CODTECH INTERNSHIP - TASK 1: DATA PIPELINE DEVELOPMENT
# Author  : [Your Name]
# Dataset : Titanic (loaded directly from a public URL)
# Tools   : pandas, scikit-learn
# Goal    : Automate the ETL (Extract → Transform → Load) process
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# ──────────────────────────────────────────────
# STEP 1: EXTRACT  – Load raw data from source
# ──────────────────────────────────────────────
def extract_data(url: str) -> pd.DataFrame:
    """
    Extract: Download the Titanic CSV directly from a public URL.
    In a real project this could be a database, API, or local file.
    """
    print("📥  [EXTRACT] Loading data from URL ...")
    df = pd.read_csv(url)
    print(f"    ✅  Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


# ──────────────────────────────────────────────
# STEP 2: TRANSFORM – Clean and engineer features
# ──────────────────────────────────────────────
def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform:
      1. Drop columns that are not useful for modelling
      2. Create a new feature (FamilySize)
      3. Separate features (X) from the target label (y)
      4. Apply a ColumnTransformer pipeline:
            • Numeric columns  → impute median  → standard-scale
            • Categorical cols → impute mode    → one-hot-encode
    Returns processed feature matrix X_processed and target series y.
    """
    print("🔧  [TRANSFORM] Cleaning and engineering features ...")

    # --- 2a. Drop low-value / leaky columns ---------------------------------
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    print(f"    Dropped columns: {columns_to_drop}")

    # --- 2b. Feature engineering: combine SibSp + Parch into FamilySize ----
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1   # +1 for the passenger
    df = df.drop(columns=["SibSp", "Parch"])
    print("    Created new feature: FamilySize = SibSp + Parch + 1")

    # --- 2c. Separate features and target -----------------------------------
    target_col = "Survived"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"    Target column : '{target_col}'")
    print(f"    Feature shape : {X.shape}")

    # --- 2d. Identify column types ------------------------------------------
    numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"    Numeric features     : {numeric_features}")
    print(f"    Categorical features : {categorical_features}")

    # --- 2e. Build sub-pipelines for each type ------------------------------
    # Numeric pipeline: fill missing values with the median, then scale
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # Categorical pipeline: fill missing values with the most frequent value,
    # then one-hot-encode (drop first category to avoid multicollinearity)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    # Combine both pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Fit & apply the preprocessor
    X_processed_array = preprocessor.fit_transform(X)

    # Rebuild as a DataFrame with readable column names
    ohe_feature_names = (
        preprocessor
        .named_transformers_["cat"]["encoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    all_feature_names = numeric_features + ohe_feature_names
    X_processed = pd.DataFrame(X_processed_array, columns=all_feature_names)

    print(f"\n    ✅  Transformed shape: {X_processed.shape}")
    print(f"    Final feature names: {X_processed.columns.tolist()}\n")
    return X_processed, y, preprocessor


# ──────────────────────────────────────────────
# STEP 3: LOAD – Save processed data to CSV
# ──────────────────────────────────────────────
def load_data(X: pd.DataFrame, y: pd.Series, output_path: str = "processed_titanic.csv") -> None:
    """
    Load: Merge features + target and persist to disk.
    In a real pipeline this could write to a database, data warehouse, or S3.
    """
    print("💾  [LOAD] Saving processed data ...")
    output_df = X.copy()
    output_df["Survived"] = y.values  # append the target column at the end
    output_df.to_csv(output_path, index=False)
    print(f"    ✅  Saved {output_df.shape[0]} rows → '{output_path}'\n")


# ──────────────────────────────────────────────
# STEP 4: SUMMARY – Quick stats on output data
# ──────────────────────────────────────────────
def print_summary(X: pd.DataFrame, y: pd.Series) -> None:
    """Print a brief summary of the processed dataset."""
    print("📊  [SUMMARY] Processed dataset overview")
    print(f"    Rows     : {X.shape[0]}")
    print(f"    Features : {X.shape[1]}")
    print(f"    Target distribution:\n{y.value_counts().to_string()}")
    print(f"\n    First 3 rows of processed features:")
    print(X.head(3).to_string())
    print("\n" + "="*60)
    print("  ETL pipeline completed successfully! 🎉")
    print("="*60)


# ──────────────────────────────────────────────
# MAIN  – Wire the ETL steps together
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Public Titanic CSV (Stanford mirror — no login needed)
    DATA_URL    = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    OUTPUT_PATH = "processed_titanic.csv"

    # Run the three ETL stages
    raw_df               = extract_data(DATA_URL)
    X_clean, y, pipeline = transform_data(raw_df)
    load_data(X_clean, y, OUTPUT_PATH)
    print_summary(X_clean, y)
