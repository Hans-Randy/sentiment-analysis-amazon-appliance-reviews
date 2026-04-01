import pandas as pd

from src.data_prep import combine_review_text, label_from_rating, prepare_dataset


def test_label_from_rating_mapping() -> None:
    assert label_from_rating(1) == "Negative"
    assert label_from_rating(2) == "Negative"
    assert label_from_rating(3) == "Neutral"
    assert label_from_rating(4) == "Positive"
    assert label_from_rating(5) == "Positive"


def test_prepare_dataset_removes_exact_duplicates_and_empty_reviews() -> None:
    raw = pd.DataFrame(
        [
            {
                "reviewerID": "r1",
                "asin": "a1",
                "summary": "Good",
                "reviewText": "Works well",
                "overall": 5,
            },
            {
                "reviewerID": "r1",
                "asin": "a1",
                "summary": "Good",
                "reviewText": "Works well",
                "overall": 5,
            },
            {
                "reviewerID": "r2",
                "asin": "a2",
                "summary": "",
                "reviewText": "   ",
                "overall": 3,
            },
            {
                "reviewerID": "r3",
                "asin": "a3",
                "summary": "Bad",
                "reviewText": "Broke quickly",
                "overall": 1,
            },
        ]
    )

    prepared = prepare_dataset(raw)

    assert len(prepared) == 2
    assert prepared["label"].tolist() == ["Positive", "Negative"]
    assert prepared["text"].tolist() == ["Good. Works well", "Bad. Broke quickly"]


def test_combine_review_text_handles_missing_values() -> None:
    assert combine_review_text("Summary", "Body") == "Summary. Body"
    assert combine_review_text("", "Body") == "Body"
    assert combine_review_text("Summary", "") == "Summary"
