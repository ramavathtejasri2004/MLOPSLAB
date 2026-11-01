
import os
import sys
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# =====================================================
# BOOKS DATASET - LAB 7 AUTOMATED MLOPS PIPELINE
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

os.chdir(PROJECT_ROOT)

DATA_PATH = "data/processed/books_preprocessed.csv"
MODEL_NAME = "BooksDatasetClassifier"
TRACKING_URI = "sqlite:///mlflow.db"

REPORT_DIR = "artifacts"
os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    REPORT_DIR,
    "pipeline_reproducibility_report.json"
)

results = []

def record_step(name, status, details=""):
    results.append({
        "step": name,
        "status": status,
        "details": details
    })
    print(f"[{status}] {name}: {details}")


print("=" * 60)
print("BOOKS DATASET - LAB 7 PIPELINE")
print("=" * 60)

start_time = datetime.now()

try:
    # 1. DATASET CHECK
    print("\n1. Checking processed dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    record_step(
        "Dataset Loading",
        "PASSED",
        f"Loaded {len(df)} rows"
    )

    # 2. DATA VALIDATION
    print("\n2. Validating dataset...")

    required_columns = [
        "book_id",
        "Book-Title",
        "Book-Author",
        "Year-Of-Publication",
        "Publisher",
        "rating_label"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df["rating_label"].isna().any():
        raise ValueError("Missing target values.")

    if not set(df["rating_label"].unique()).issubset({0, 1}):
        raise ValueError("Invalid target labels.")

    record_step(
        "Data Validation",
        "PASSED",
        "Required columns and target verified"
    )

    # 3. DATA QUALITY
    print("\n3. Checking data quality...")

    duplicate_count = int(df.duplicated().sum())
    missing_values = int(df.isnull().sum().sum())

    record_step(
        "Data Quality",
        "PASSED",
        f"Duplicates={duplicate_count}, Missing values={missing_values}"
    )

    # 4. PREPROCESSING VERIFICATION
    print("\n4. Checking preprocessing output...")

    if "book_id" in df.columns:
        unique_books = int(df["book_id"].nunique())
    else:
        unique_books = 0

    class_distribution = {
        str(k): int(v)
        for k, v in df["rating_label"].value_counts().items()
    }

    record_step(
        "Preprocessing Verification",
        "PASSED",
        f"Rows={len(df)}, Unique books={unique_books}"
    )

    # 5. MLFLOW MODEL REGISTRY
    print("\n5. Checking MLflow model registry...")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    registered_model = client.get_registered_model(
        MODEL_NAME
    )

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        "candidate"
    )

    record_step(
        "MLflow Registry",
        "PASSED",
        f"Model={registered_model.name}, Candidate version={model_version.version}"
    )

    # 6. LOAD REGISTERED MODEL
    print("\n6. Loading registered model...")

    model_uri = f"models:/{MODEL_NAME}@candidate"

    model = mlflow.sklearn.load_model(model_uri)

    record_step(
        "Model Loading",
        "PASSED",
        f"Loaded {model_uri}"
    )

    # 7. TEST INFERENCE
    print("\n7. Testing inference...")

    features = [
        "Book-Title",
        "Book-Author",
        "Year-Of-Publication",
        "Publisher"
    ]

    sample = df[features].head(5)

    predictions = model.predict(sample)

    record_step(
        "Inference",
        "PASSED",
        f"Generated {len(predictions)} predictions"
    )

    # 8. SAVE REPORT
    print("\n8. Saving pipeline report...")

    end_time = datetime.now()

    report = {
        "project": "Books Dataset MLOps",
        "lab": "Lab 7",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (
            end_time - start_time
        ).total_seconds(),
        "dataset_rows": len(df),
        "unique_books": unique_books,
        "class_distribution": class_distribution,
        "model_name": MODEL_NAME,
        "model_version": model_version.version,
        "predictions": predictions.tolist(),
        "steps": results,
        "final_status": "PASSED"
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print("\nReport saved:", REPORT_PATH)

    print("\n" + "=" * 60)
    print("LAB 7 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

except Exception as error:

    record_step(
        "Pipeline Execution",
        "FAILED",
        str(error)
    )

    report = {
        "project": "Books Dataset MLOps",
        "lab": "Lab 7",
        "steps": results,
        "final_status": "FAILED",
        "error": str(error)
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nPipeline failed:", error)
    sys.exit(1)
