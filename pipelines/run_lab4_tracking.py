import os
import pandas as pd
import mlflow


# ============================================================
# 1. MLflow configuration
# ============================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")


mlflow.set_experiment("Books_Dataset_Classification")


# ============================================================
# 2. Load existing results
# ============================================================

results_file = "outputs/model_comparison.csv"

results = pd.read_csv(results_file)

print("Loaded model comparison results:")
print(results)


# ============================================================
# 3. Track each trained model
# ============================================================

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "SVM": "models/svm.pkl",
    "KNN": "models/knn.pkl"
}


for _, row in results.iterrows():

    model_name = row["Model"]

    print("\n" + "=" * 60)
    print("Tracking:", model_name)
    print("=" * 60)

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "dataset",
            "Books Dataset"
        )

        mlflow.log_param(
            "split_strategy",
            "StratifiedGroupKFold"
        )

        mlflow.log_param(
            "test_size_approx",
            0.20
        )

        mlflow.log_param(
            "group_column",
            "book_id"
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            float(row["Accuracy"])
        )

        mlflow.log_metric(
            "precision",
            float(row["Precision"])
        )

        mlflow.log_metric(
            "recall",
            float(row["Recall"])
        )

        mlflow.log_metric(
            "f1_score",
            float(row["F1 Score"])
        )

        mlflow.log_metric(
            "training_time_seconds",
            float(row["Training Time"])
        )

        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        mlflow.log_param(
            "dataset_rows",
            226007
        )

        mlflow.log_param(
            "unique_books",
            138483
        )

        mlflow.log_param(
            "features",
            "Book-Title, Book-Author, Year-Of-Publication, Publisher"
        )

        # ----------------------------------------------------
        # Log model file
        # ----------------------------------------------------

        model_path = model_files.get(model_name)

        if model_path and os.path.exists(model_path):

            mlflow.log_artifact(
                model_path,
                artifact_path="model"
            )

            print(
                "Model artifact logged:",
                model_path
            )

        print("Run ID:", mlflow.active_run().info.run_id)

    print("Tracking completed.")


# ============================================================
# 4. Log comparison CSV as an artifact
# ============================================================

with mlflow.start_run(
    run_name="Model_Comparison"
):

    mlflow.log_artifact(
        results_file,
        artifact_path="results"
    )

    mlflow.log_param(
        "experiment_type",
        "model_comparison"
    )

    print("\nComparison results logged to MLflow.")


print("\n" + "=" * 60)
print("MLFLOW TRACKING COMPLETED")
print("=" * 60)

print("\nExperiment:")
print("Books_Dataset_Classification")

print("\nTracking directory:")
print("mlruns")