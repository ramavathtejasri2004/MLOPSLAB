from pathlib import Path
import cloudpickle
import pandas as pd

from fastapi import FastAPI, HTTPException
from api.schemas import BookRequest


# Load exported ML model
MODEL_PATH = Path(__file__).resolve().parent.parent / "deployment" / "model" / "model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = cloudpickle.load(file)


# Create FastAPI application
app = FastAPI(
    title="Books Dataset Prediction API",
    description="Predict whether a book is Like or Dislike.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Books Dataset Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_book(book: BookRequest):

    try:
        input_data = pd.DataFrame([
            book.model_dump(by_alias=True)
        ])

        prediction = int(model.predict(input_data)[0])

        recommendation = "Like" if prediction == 1 else "Dislike"

        return {
            "book_title": book.book_title,
            "author": book.book_author,
            "predicted_class": prediction,
            "recommendation": recommendation
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
