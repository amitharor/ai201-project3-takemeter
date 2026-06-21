# Baseline prompt (Groq zero shot): notebook Section 5

This is the exact prompt used for the **zero shot baseline**: `llama-3.3-70b-versatile`
classifying each test comment with no task specific training. It is pasted into the notebook's
Section 5 `SYSTEM_PROMPT` cell and used as a **system message**; the notebook passes each comment
as a separate **user message**. The model outputs only the label name, which the notebook parses.

The label definitions are taken from `planning.md §2`, with **one illustrative example per
label**. The examples come from the §2 taxonomy, **not** from the test set, so the baseline sees
no test data.

***

## The prompt

```
You are classifying comments from r/nba (a basketball subreddit) by HOW each comment supports its claim, not by topic and not by whether the claim is correct. Assign each comment to exactly one of these three categories.

analysis: A structured argument backed by specific, verifiable evidence (stats, historical comparison, tactical or film observation, or a clear causal chain). If you strip the opinion framing, a real argument remains.
Example: "Their half court defensive rating is 3rd since the All Star break; the regression is entirely in transition (12th to 27th). That is effort, not scheme."

hot_take: A bold, confident opinion asserted without genuine supporting evidence. The claim may be correct, but it asserts rather than argues. A decorative or cherry picked stat that only sounds credible still counts.
Example: "Jokic is the most skilled big man to ever touch a basketball and it is not close. Anyone who disagrees does not watch basketball."

reaction: An immediate emotional response to a play, game, or event. Little to no argument; expressing a feeling in the moment (hype, despair, shock, humor).
Example: "BANG. game over. I am levitating right now."

Rules:
Output ONLY the label name, one lowercase word, nothing else.
No punctuation, no explanation, no quotes.
If a comment is emotional with only a throwaway justification, it is reaction.
If a comment is long but gives no verifiable evidence, it is hot_take (length is not evidence).

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction
```

***

## Notes for running it
* **Temperature 0** for determinism and reproducibility.
* **max_tokens** small (about 4): only one word is needed.
* All **45 of 45** test responses parsed cleanly (0% unparseable), so no tightening was needed.

## What the baseline showed (planning.md §5 hypothesis, revisited)
The §5 hypothesis was that the baseline would do fine on `reaction` but struggle on the
`analysis` versus `hot_take` boundary. The result was the opposite of that second half: the 70B
scored F1 0.85 on `analysis` and 0.81 on `hot_take`, handling that subjective boundary well. It
was the fine tuned DistilBERT that failed there (F1 0.44 and 0.42). So the hard boundary is hard
for a small model trained on a few hundred examples, not for a large general model with good
label definitions. That contrast is the core of the evaluation report.
