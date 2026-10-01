import pandas as pd
import os

INPUT_FILE = "data/processed/final_books_dataset.csv"
OUTPUT_FILE = "data/processed/books_preprocessed.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, dtype=str)

print("Original shape:", df.shape)

# Convert rating to numeric
df["Book-Rating"] = pd.to_numeric(
    df["Book-Rating"].astype(str).str.strip().str.replace('"', ''),
    errors="coerce"
)

# Remove rating 0
df = df[df["Book-Rating"] > 0].copy()

print("Shape after removing rating 0:", df.shape)

# Create binary target
# 1-6  -> Dislike (0)
# 7-10 -> Like (1)
df["rating_label"] = (df["Book-Rating"] >= 7).astype(int)

# Keep ISBN only as a GROUP identifier.
# It will NOT be used as a machine-learning feature.
df["book_id"] = (
    df["ISBN"]
    .astype(str)
    .str.strip()
    .str.replace('"', '', regex=False)
)

# Clean text columns
text_columns = [
    "Book-Title",
    "Book-Author",
    "Publisher"
]

for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
    )

# Convert publication year
df["Year-Of-Publication"] = pd.to_numeric(
    df["Year-Of-Publication"]
    .astype(str)
    .str.strip()
    .str.replace('"', ''),
    errors="coerce"
)

# Invalid years
df.loc[
    (df["Year-Of-Publication"] < 1000) |
    (df["Year-Of-Publication"] > 2026),
    "Year-Of-Publication"
] = pd.NA

# Fill missing years with median
median_year = df["Year-Of-Publication"].median()

df["Year-Of-Publication"] = df[
    "Year-Of-Publication"
].fillna(median_year)

# Remove exact duplicate rows
df = df.drop_duplicates()

# Keep only required columns
df = df[
    [
        "book_id",
        "Book-Title",
        "Book-Author",
        "Year-Of-Publication",
        "Publisher",
        "rating_label"
    ]
]

print("\nPreprocessing completed!")
print("Final shape:", df.shape)

print("\nTarget distribution:")
print(df["rating_label"].value_counts())

print("\nNumber of unique books:")
print(df["book_id"].nunique())

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:")
print(OUTPUT_FILE)