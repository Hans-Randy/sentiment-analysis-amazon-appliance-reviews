# Section 15 - Rating Enhancement

## Method Name

Overall-opinion-enhanced rating blend.

## Why This Method Was Chosen

This project brief explicitly asks for a review-based enhancement of rating values. A simplified overall-opinion method is the best fit for this repo because it reuses the strongest existing sentiment model, works directly with the available review text, and avoids the extra complexity of topic, helpfulness, or aspect pipelines that are not already implemented here.

## Paper-Inspired Approach

The method estimates an inferred opinion from the review text, maps that opinion to the same 1-5 rating scale as the original Amazon rating, then blends the two values with a tunable alpha. This follows the brief's simplified interpretation of the paper idea that overall opinions from reviews can refine explicit ratings.

## ASCII Workflow Diagram

```text
review text + original rating
        |
        v
 preprocess / select usable text
        |
        v
 infer overall opinion with trained sentiment model
        |
        v
 map opinion to inferred 1-5 rating
        |
        v
 blend with original rating using alpha
        |
        v
 evaluate MAE and RMSE across alpha values
```

## Pseudo-code

```text
FOR each review:
    preprocess text
    infer overall opinion
    map opinion to inferred rating
    combine inferred rating with original rating using alpha
END FOR
evaluate multiple alpha values
select and discuss the best setting
```

## Assumptions

- The prepared large Phase 2 dataset is the correct source for this section.
- The merged `text` field, built from `summary` and `reviewText`, is the best available field for overall-opinion inference.
- The best existing model artifact available in the repo is `phase2_mlp.joblib`.
- The original `overall` rating is treated as the evaluation reference because no external gold enhanced-rating target exists.
- Sentiment labels are mapped to the 1-5 scale with anchors: Negative -> 1.5, Neutral -> 3.0, Positive -> 4.5.
- When the model exposes class probabilities, the inferred numeric rating is the probability-weighted expected rating on this 1-5 anchor scale.

## Preprocessing Assumptions

- Rows with missing or empty review text are already removed in the prepared dataset.
- Exact duplicate review rows are already removed by the existing Phase 2 preprocessing pipeline.
- The merged `text` field is reused as-is to keep this section consistent with the sentiment models already trained in the repo.

## Usable Row Count

- 591015 reviews

## Sentiment-to-Rating Mapping

- Negative -> 1.5
- Neutral -> 3.0
- Positive -> 4.5

## Results

### Alpha Evaluation

| alpha | mae | rmse | enhanced_mean | enhanced_std |
| --- | --- | --- | --- | --- |
| 0.0 | 0.6664 | 0.8106 | 4.0311 | 0.8959 |
| 0.9 | 0.0666 | 0.0811 | 4.2501 | 1.2462 |
| 0.8 | 0.1333 | 0.1621 | 4.2258 | 1.1924 |
| 0.7 | 0.1999 | 0.2432 | 4.2014 | 1.1413 |
| 0.6 | 0.2666 | 0.3243 | 4.1771 | 1.0933 |

### Distribution Comparison Using Best Alpha (0.9)

| Rating | Original | Inferred | Enhanced |
| --- | ---: | ---: | ---: |
| 1.0 | 58579 | 0 | 58579 |
| 2.0 | 20389 | 68291 | 20389 |
| 3.0 | 29744 | 31456 | 29744 |
| 4.0 | 73848 | 491268 | 73848 |
| 5.0 | 408455 | 0 | 408455 |

### Example Rows

| dataset_index | text_excerpt | original_rating | inferred_label | inferred_rating | enhanced_rating |
| --- | --- | --- | --- | --- | --- |
| 279832 | Perfect fit. Most Excellent Product and Service. Perfect fit. Most Excellent Product and Service!!! Thank You | 1.0 | Positive | 4.5 | 1.35 |
| 111111 | Works perfectly and I am very pleased. And the cost is awesome compared to the .... Works perfectly and I am very pleased. And the cost is awesome compared to the name brand!!! ... | 1.0 | Positive | 4.5 | 1.35 |
| 171334 | Worked Great!. Worked like a charm. Saved a lot of money although I had to call a third party to install it. Still saved money on parts. | 1.0 | Positive | 4.5 | 1.35 |
| 565333 | OUCH. WORKS VERY NICE. WIFE IS PLEASED. EASY YO CLEAN ETC. FAILED AFTER 2 YEARS. GET WARRANTEE. | 1.0 | Positive | 4.5 | 1.35 |
| 585567 | These are great, have one in every room. These are great, have one in every room. I can keep my thetmostat a little higher, and be confident that individual rooms are a few degr... | 1.0 | Positive | 4.5 | 1.35 |

## Conclusion

The best alpha on this evaluation setup is 0.9, which is expected because the original rating is used as the reference target. Even so, the experiment demonstrates a reproducible way to incorporate overall opinion inferred from review text into a blended rating value, which satisfies the project brief's rating-enhancement requirement.
