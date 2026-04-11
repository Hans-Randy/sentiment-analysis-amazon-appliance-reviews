import pandas as pd
from typing import cast

from src.config import REPORTS_DIR, SECTION16_SUMMARY_COUNT, TABLES_DIR
from src.data_utils import (
    load_phase2_reviews,
    select_long_reviews,
    text_excerpt,
    word_count,
)
from src.hf_utils import (
    DEFAULT_SUMMARIZATION_MODEL,
    generate_text,
    load_local_seq2seq,
)
from src.utils import ensure_directories, markdown_table_from_rows, write_markdown


def summarization_prompt(review_text: str) -> str:
    return (
        "Summarize the following customer review in 2 sentences and roughly 45 to 60 words. "
        "Keep the main product opinion, major issue or benefit, and any important context.\n\n"
        f"Review:\n{review_text}"
    )


def summarize_reviews(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    seq2seq_model = load_local_seq2seq(model_name)
    rows = []
    for _, row in df.iterrows():
        review_text = str(row["text"])
        summary = generate_text(
            seq2seq_model,
            summarization_prompt(review_text),
            max_new_tokens=96,
            min_new_tokens=45,
        )
        rows.append(
            {
                "dataset_index": int(cast(int, row["dataset_index"])),
                "asin": str(row["asin"]),
                "overall": float(cast(float, row["overall"])),
                "original_word_count": int(cast(int, row["review_word_count"])),
                "summary_word_count": word_count(summary),
                "original_excerpt": text_excerpt(review_text, max_chars=220),
                "generated_summary": summary,
                "model_name": model_name,
            }
        )
    return pd.DataFrame(rows)


def build_report(results_df: pd.DataFrame, model_name: str) -> str:
    summary_table = markdown_table_from_rows(
        ["dataset_index", "original_word_count", "summary_word_count"],
        results_df[
            ["dataset_index", "original_word_count", "summary_word_count"]
        ].values.tolist(),
    )
    first_two = results_df.head(2)
    example_sections = []
    for example_number, (_, row) in enumerate(first_two.iterrows(), start=1):
        example_sections.append(
            f"## Example {example_number}\n\n"
            f"Original review (dataset index {int(cast(int, row['dataset_index']))}):\n\n"
            f"```text\n{row['original_full_text']}\n```\n\n"
            f"Generated summary:\n\n"
            f"```text\n{row['generated_summary']}\n```\n"
        )
    examples_text = "\n".join(example_sections)
    return f"""# Section 16 - Local LLM Summarization

## Model Used

- `{model_name}`

## Why This Model Is Suitable

This model is a compact sequence-to-sequence summarization model that can run locally on CPU hardware. It is suitable for summarization because it is specifically fine-tuned for summarization-style generation and can produce concise review summaries without requiring a remote API.

## Prompt / Generation Setup

- Task format: instruction-style summarization prompt
- Target length: about 50 words
- Decoding: deterministic (`do_sample=False`)
- `min_new_tokens=45`
- `max_new_tokens=96`

## Selected Reviews Summary Table

{summary_table}

{examples_text}

## Discussion

The summaries generally preserve the overall product opinion and the main issue or benefit. Strengths include concise extraction of customer intent and easy reproducibility. Limitations include occasional wording stiffness and imperfect control over exact summary length because the local model is intentionally small and CPU-friendly.
"""


def main() -> None:
    ensure_directories([TABLES_DIR, REPORTS_DIR])
    reviews_df = load_phase2_reviews()
    selected_df = select_long_reviews(reviews_df, count=SECTION16_SUMMARY_COUNT)
    results_df = summarize_reviews(selected_df, DEFAULT_SUMMARIZATION_MODEL)
    original_text_df = selected_df[["dataset_index", "text"]].copy()
    original_text_df.columns = ["dataset_index", "original_full_text"]
    results_df = results_df.merge(
        original_text_df,
        on="dataset_index",
        how="left",
    )
    results_df.to_csv(TABLES_DIR / "section16_review_summaries.csv", index=False)
    write_markdown(
        build_report(results_df, DEFAULT_SUMMARIZATION_MODEL),
        REPORTS_DIR / "section16_llm_summarization.md",
    )

    print("Section 16 summarization complete.")
    print(
        results_df[
            ["dataset_index", "original_word_count", "summary_word_count"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
