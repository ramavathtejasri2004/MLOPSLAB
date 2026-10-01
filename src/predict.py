import os
import mlflow
import pandas as pd


# ============================================================
# BOOKS DATASET - MODEL PREDICTION
# ============================================================

MODEL_NAME = "BooksDatasetClassifier"
MODEL_VERSION = "1"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_result.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD REGISTERED MODEL
# ============================================================

print("=" * 60)
print("BOOKS DATASET MODEL PREDICTION")
print("=" * 60)

print("\nLoading registered model...")
print("Model:", MODEL_NAME)
print("Version:", MODEL_VERSION)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

try:

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print("Model loaded successfully.")

except Exception as e:

    print("\nERROR: Could not load model.")
    print(e)
    raise


# ============================================================
# GET BOOK DETAILS
# ============================================================

print("\nEnter book details")
print("-" * 60)

book_title = input("Book Title: ").strip()

book_author = input("Book Author: ").strip()

year = input(
    "Year of Publication: "
).strip()

publisher = input(
    "Publisher: "
).strip()


# ============================================================
# VALIDATE YEAR
# ============================================================

try:

    year = float(year)

except ValueError:

    print(
        "\nInvalid year. Please enter a numeric year."
    )

    raise SystemExit


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([
    {
        "Book-Title": book_title,
        "Book-Author": book_author,
        "Year-Of-Publication": year,
        "Publisher": publisher
    }
])


print("\nInput data:")
print(input_data)


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(
    input_data
)[0]


# ============================================================
# CONVERT LABEL
# ============================================================

if prediction == 1:

    result = "Like"

else:

    result = "Dislike"


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("Book Title :", book_title)
print("Author     :", book_author)
print("Year       :", year)
print("Publisher  :", publisher)

print("\nPredicted Class :", int(prediction))
print("Recommendation  :", result)

print("=" * 60)


# ============================================================
# SAVE RESULT
# ============================================================

result_data = pd.DataFrame([
    {
        "Book-Title": book_title,
        "Book-Author": book_author,
        "Year-Of-Publication": year,
        "Publisher": publisher,
        "Predicted-Class": int(prediction),
        "Recommendation": result
    }
])

result_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPrediction saved to:")
print(OUTPUT_FILE)