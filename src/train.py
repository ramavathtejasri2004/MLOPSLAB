import pandas as pd
import os
import time
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "data/processed/books_preprocessed.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET, FEATURES AND GROUPS
# ============================================================

target = "rating_label"
group_column = "book_id"

y = df[target]

# book_id is used ONLY for grouping.
# It must NOT be used as a machine-learning feature.
groups = df[group_column]

X = df.drop(columns=[target, group_column])

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

print("\nCreating group-aware train/test split...")

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    sgkf.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 4. CHECK FOR BOOK LEAKAGE
# ============================================================

common_books = set(groups_train).intersection(set(groups_test))

print("\nCommon books between train and test:", len(common_books))

if len(common_books) == 0:
    print("✓ No book leakage detected.")
else:
    print("WARNING: Book leakage detected!")


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_features = [
    "Year-Of-Publication"
]

categorical_features = [
    "Book-Title",
    "Book-Author",
    "Publisher"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "SVM":
        SVC(
            kernel="rbf"
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
}


# ============================================================
# 7. CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 8. TRAIN AND EVALUATE
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    start_time = time.time()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training...")

    pipeline.fit(X_train, y_train)

    print("Predicting...")

    y_pred = pipeline.predict(X_test)

    training_time = time.time() - start_time

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTraining Time:", round(training_time, 2), "seconds")

    # Save model
    filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = os.path.join(
        "models",
        filename
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print("Model saved:", model_path)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time": training_time
    })


# ============================================================
# 9. SAVE COMPARISON RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print("outputs/model_comparison.csv")

print("\nTraining completed successfully!")