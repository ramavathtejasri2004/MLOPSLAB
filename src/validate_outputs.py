import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# ============================================================
# CONFIGURATION
# ============================================================

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "BooksDatasetClassifier"
MODEL_VERSION = "1"

DATA_PATH = "data/processed/books_preprocessed.csv"
OUTPUT_DIR = "reports"
REPORT_PATH = os.path.join(OUTPUT_DIR, "model_validation.txt")

# ============================================================
# SETUP
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("REGISTERED MODEL VALIDATION")
print("=" * 60)

# ============================================================
# 1. CHECK DATASET
# ============================================================

print("\n[1] Checking dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset found.")
print("Dataset shape:", df.shape)

# ============================================================
# 2. CHECK REGISTERED MODEL
# ============================================================

print("\n[2] Checking MLflow Model Registry...")

try:
    registered_model = client.get_registered_model(MODEL_NAME)

    print("Registered model found:")
    print("Model name:", registered_model.name)

except Exception as e:
    raise RuntimeError(
        f"Registered model '{MODEL_NAME}' was not found."
    ) from e

# ============================================================
# 3. CHECK MODEL VERSION
# ============================================================

print("\n[3] Checking model version...")

try:
    model_version = client.get_model_version(
        name=MODEL_NAME,
        version=MODEL_VERSION
    )

    print("Model version found.")
    print("Version:", model_version.version)
    print("Run ID:", model_version.run_id)
    print("Status:", model_version.status)

except Exception as e:
    raise RuntimeError(
        f"Model version {MODEL_VERSION} was not found."
    ) from e

# ============================================================
# 4. CHECK CANDIDATE ALIAS
# ============================================================

print("\n[4] Checking candidate alias...")

try:
    alias_version = client.get_model_version_by_alias(
        MODEL_NAME,
        "candidate"
    )

    print("Candidate alias found.")
    print("Candidate version:", alias_version.version)

except Exception as e:
    print("WARNING: Candidate alias could not be read.")
    print(e)

# ============================================================
# 5. LOAD REGISTERED MODEL
# ============================================================

print("\n[5] Loading registered model...")

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)

    print("Registered model loaded successfully.")

except Exception as e:
    raise RuntimeError(
        "Could not load the registered model."
    ) from e

# ============================================================
# 6. TEST PREDICTION
# ============================================================

print("\n[6] Testing prediction...")

FEATURES = [
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher"
]

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

sample = df[FEATURES].head(5)

predictions = model.predict(sample)

print("\nSample input:")
print(sample)

print("\nPredictions:")
print(predictions)

print("\nPrediction test successful.")

# ============================================================
# 7. SAVE VALIDATION REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as file:

    file.write("BOOKS DATASET ML MODEL VALIDATION REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Model Name       : {MODEL_NAME}\n")
    file.write(f"Model Version    : {MODEL_VERSION}\n")
    file.write(f"Run ID           : {model_version.run_id}\n")
    file.write(f"Model Status     : {model_version.status}\n")
    file.write(f"Dataset Shape    : {df.shape}\n")
    file.write(f"Model URI        : {MODEL_URI}\n")
    file.write("\nFeatures:\n")

    for feature in FEATURES:
        file.write(f"- {feature}\n")

    file.write("\nSample Predictions:\n")

    for prediction in predictions:
        file.write(f"{prediction}\n")

    file.write("\nValidation Result: PASSED\n")

print("\n" + "=" * 60)
print("VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nReport saved to:")
print(REPORT_PATH)