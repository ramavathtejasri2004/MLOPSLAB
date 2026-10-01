import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Books_Dataset_Classification"
MODEL_NAME = "BooksDatasetClassifier"

MODEL_PATH = "models/svm.pkl"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

print("=" * 60)
print("MODEL REGISTRY - BOOKS DATASET")
print("=" * 60)

# --------------------------------------------------
# 1. Check model file
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print(f"\nModel file found: {MODEL_PATH}")

# --------------------------------------------------
# 2. Get experiment
# --------------------------------------------------

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' not found."
    )

experiment_id = experiment.experiment_id

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Experiment ID: {experiment_id}")

# --------------------------------------------------
# 3. Find existing SVM MLflow run
# --------------------------------------------------

runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="params.model = 'SVM'",
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No SVM MLflow run found.")

run = runs[0]
run_id = run.info.run_id

print(f"\nExisting SVM Run ID:")
print(run_id)

# --------------------------------------------------
# 4. Load trained SVM model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("\nTrained SVM model loaded successfully.")

# --------------------------------------------------
# 5. Log SVM as an MLflow model
# --------------------------------------------------

print("\nLogging SVM model into MLflow...")

with mlflow.start_run(run_id=run_id):
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="sklearn_model",
        skops_trusted_types=[
            "scipy.sparse._csr.csr_matrix"
        ]
    )

    

model_uri = model_info.model_uri

print("MLflow model logged successfully.")
print(f"Model URI: {model_uri}")

# --------------------------------------------------
# 6. Register model
# --------------------------------------------------

print("\nRegistering model...")

try:
    client.get_registered_model(MODEL_NAME)
    print(f"Registered model already exists: {MODEL_NAME}")

except Exception:
    client.create_registered_model(
        MODEL_NAME,
        description=(
            "Books Dataset Classification model "
            "using SVM with group-aware evaluation."
        )
    )
    print(f"Created registered model: {MODEL_NAME}")

# --------------------------------------------------
# 7. Create model version
# --------------------------------------------------

registered_version = client.create_model_version(
    name=MODEL_NAME,
    source=model_uri,
    run_id=run_id
)

print("\nModel registered successfully!")

print(f"Model Name    : {MODEL_NAME}")
print(f"Model Version : {registered_version.version}")
print(f"Run ID        : {run_id}")

# --------------------------------------------------
# 8. Add model metadata
# --------------------------------------------------

client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered_version.version,
    key="model_type",
    value="SVM"
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered_version.version,
    key="selection_metric",
    value="F1 Score"
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered_version.version,
    key="dataset",
    value="Books Dataset"
)

print("\nModel metadata added successfully.")

# --------------------------------------------------
# 9. Create alias
# --------------------------------------------------

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="candidate",
    version=registered_version.version
)

print(
    f"\nAlias 'candidate' assigned to version "
    f"{registered_version.version}"
)

# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL REGISTRY COMPLETED")
print("=" * 60)