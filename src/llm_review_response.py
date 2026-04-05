import pandas as pd
from typing import cast

from src.config import REPORTS_DIR, TABLES_DIR
from src.data_utils import (
    load_phase2_reviews,
    select_question_like_review,
    text_excerpt,
)
from src.hf_utils import (
    DEFAULT_RESPONSE_MODEL,
    generate_chat_response,
    load_local_causal,
)
from src.utils import ensure_directories, write_markdown


def response_prompt(review_text: str) -> str:
    return (
        "You write short customer service replies for appliance reviews.\n\n"
        "Example review:\n"
        "The handle arrived cracked. Should I return it or can I replace just the handle?\n\n"
        "Example response:\n"
        "Thank you for letting us know about the cracked handle. Please contact the seller or check the product warranty to see whether a replacement part or return is available for your order.\n\n"
        "Now write one polite, helpful, concise response for this review. "
        "Acknowledge the issue and suggest a safe next step. Do not make guarantees.\n\n"
        f"Review:\n{review_text}\n\nResponse:"
    )


def build_selection_reason(row: pd.Series) -> str:
    reasons = []
    text = str(row["text"]).lower()
    if "?" in text:
        reasons.append("contains a question mark")
    for phrase in [
        "does this",
        "can i",
        "is this",
        "will this",
        "how do i",
        "should i",
    ]:
        if phrase in text:
            reasons.append(f"contains the phrase '{phrase}'")
    reasons.append(
        f"heuristic question score = {int(cast(int, row['question_score']))}"
    )
    return "; ".join(reasons)


def build_report(result_row: pd.Series) -> str:
    return f"""# Section 17 - Local LLM Customer Response

## Selected Review

Dataset index: `{int(cast(int, result_row["dataset_index"]))}`

> {result_row["review_text"]}

## Why It Qualified As Question-Like

- {result_row["selection_reason"]}

## Model Used

- `{result_row["model_name"]}`

## Prompt Template

```text
You write short customer service replies for appliance reviews.

Example review:
The handle arrived cracked. Should I return it or can I replace just the handle?

Example response:
Thank you for letting us know about the cracked handle. Please contact the seller or check the product warranty to see whether a replacement part or return is available for your order.

Now write one polite, helpful, concise response for this review.
Review:
<review text>

Response:
```

## Generation Settings

- deterministic decoding (`do_sample=False`)
- causal local chat model
- `max_new_tokens=120`

## Generated Response

> {result_row["generated_response"]}

## Discussion

The response is designed to sound professional, acknowledge the customer's concern, and stay close to the information contained in the review. The main limitation is that the local CPU-friendly model can produce generic wording and should not be treated as an official product support channel without human review.
"""


def main() -> None:
    ensure_directories([TABLES_DIR, REPORTS_DIR])
    reviews_df = load_phase2_reviews()
    selected_row = select_question_like_review(reviews_df)
    causal_model = load_local_causal(DEFAULT_RESPONSE_MODEL)
    review_text = str(selected_row["text"])
    response = generate_chat_response(
        causal_model,
        response_prompt(review_text),
        max_new_tokens=120,
    )

    result_df = pd.DataFrame(
        [
            {
                "dataset_index": int(cast(int, selected_row.name)),
                "asin": str(selected_row["asin"]),
                "overall": float(cast(float, selected_row["overall"])),
                "review_excerpt": text_excerpt(review_text, max_chars=220),
                "review_text": review_text,
                "selection_reason": build_selection_reason(selected_row),
                "prompt": response_prompt(review_text),
                "generated_response": response,
                "model_name": DEFAULT_RESPONSE_MODEL,
            }
        ]
    )
    result_df.to_csv(TABLES_DIR / "section17_customer_response.csv", index=False)
    write_markdown(
        build_report(result_df.iloc[0]),
        REPORTS_DIR / "section17_llm_response.md",
    )

    print("Section 17 response generation complete.")
    print(result_df[["dataset_index", "selection_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
