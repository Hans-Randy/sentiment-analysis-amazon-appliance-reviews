# Section 16 - Local LLM Summarization

## Model Used

- `sshleifer/distilbart-cnn-12-6`

## Why This Model Is Suitable

This model is a compact sequence-to-sequence summarization model that can run locally on CPU hardware. It is suitable for summarization because it is specifically fine-tuned for summarization-style generation and can produce concise review summaries without requiring a remote API.

## Prompt / Generation Setup

- Task format: instruction-style summarization prompt
- Target length: about 50 words
- Decoding: deterministic (`do_sample=False`)
- `min_new_tokens=45`
- `max_new_tokens=96`

## Selected Reviews Summary Table

| dataset_index | original_word_count | summary_word_count |
| --- | --- | --- |
| 3442 | 101 | 44 |
| 472035 | 101 | 45 |
| 17876 | 101 | 71 |
| 31923 | 101 | 45 |
| 1375 | 101 | 47 |
| 9939 | 101 | 48 |
| 463511 | 101 | 57 |
| 31957 | 101 | 49 |
| 26663 | 101 | 49 |
| 586 | 101 | 51 |

## Example 1

Original review (dataset index 3442):

```text
Well worth the money.... This little machine is awesome! Definately worth the money. When I was doing my homework on portable washers, this one seemed to be the only one I could find that had a spin cycle. (what's the point of having a washing machine if you have to hand wring the clothes?) I am single and an apartment dweller; and this little wahser is the perfect size for me and the apartment. I would definately reccommend it to my friends and family. Now if I can just find a portable electric dryer about the same size....

(PS...mine plays Jingle Bells too! Too funny!)
```

Generated summary:

```text
This little machine is awesome! It was the only one I could find that had a spin cycle . I would definately reccommend it to my friends and family. Now if I can just find a portable electric dryer about the same size .
```

## Example 2

Original review (dataset index 472035):

```text
MAYTAG STINKS. I purchased this dishwasher sixteen months ago. A couple months ago my home's power went out. When the power came back on a few minutes later, all the appliances came back on except for this crappy dishwasher. There was no storm or anything unusual, the power just went out for some reason. So I called Maytag because I'd bought an extended warranty. And they say their warranty will not cover this. What?! What house in the world NEVER has a power outage? Freaking unbelievable. Now I'll throw away the Maytag and replace with anything but. Avoid MAYTAG like the black death.
```

Generated summary:

```text
Maytag dishwasher was purchased 16 months ago . Maytag's warranty will not cover the loss of power to the dishwasher . Avoid Maytag like the black death. Avoid MAYTAG like the death. Keep the main product opinion, major issue or benefit, and important context .
```


## Discussion

The summaries generally preserve the overall product opinion and the main issue or benefit. Strengths include concise extraction of customer intent and easy reproducibility. Limitations include occasional wording stiffness and imperfect control over exact summary length because the local model is intentionally small and CPU-friendly.
