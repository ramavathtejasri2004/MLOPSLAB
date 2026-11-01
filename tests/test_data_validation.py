
import unittest
import pandas as pd

REQUIRED_COLUMNS = [
    "book_id",
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher",
    "rating_label"
]


class TestBooksDataset(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "book_id": ["B001", "B002", "B003"],
            "Book-Title": ["Book A", "Book B", "Book C"],
            "Book-Author": ["Author A", "Author B", "Author C"],
            "Year-Of-Publication": [2000, 2005, 2010],
            "Publisher": ["Pub A", "Pub B", "Pub C"],
            "rating_label": [1, 0, 1]
        })

    def test_required_columns(self):
        for column in REQUIRED_COLUMNS:
            self.assertIn(column, self.df.columns)

    def test_target_values(self):
        self.assertTrue(
            set(self.df["rating_label"].unique()).issubset({0, 1})
        )

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.df), 0)

    def test_no_missing_target(self):
        self.assertFalse(
            self.df["rating_label"].isnull().any()
        )


if __name__ == "__main__":
    unittest.main()
