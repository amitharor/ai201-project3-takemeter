# Baseline prompt (Groq zero shot): notebook Section 5

This is the exact prompt for the **zero shot baseline**: `llama-3.3-70b-versatile`
classifying each test comment with no task specific training. Paste it into the notebook's
Section 5 prompt cell. The notebook parses the response, so the model must output **only the
label name**.

The label definitions below are copied verbatim from `planning.md §2` so the baseline is
judged on the same rubric as the fine tuned model.

***

## System / instruction prompt

```
You are classifying comments from r/nba (a basketball subreddit) by HOW the comment supports
its claim, not by topic and not by whether the claim is correct.

Choose exactly ONE of these three labels:

analysis:  A structured argument backed by specific, verifiable evidence: statistics,
            historical comparison, tactical/film observation, or a clear causal chain. If you
            removed the opinion framing, a real argument would remain.

hot_take:  A bold, confident opinion asserted WITHOUT genuine supporting evidence. The claim
            may be correct, but the comment asserts rather than argues. A decorative or
            cherry picked stat that exists only to sound credible still counts as hot_take.

reaction:  An immediate emotional response to a specific play, game, or event. Little to no
            argument; the comment is expressing a feeling in the moment (hype, despair, shock,
            humor).

Rules:
* Output ONLY the label: analysis, hot_take, or reaction.
* No punctuation, no explanation, no quotes, just the single word.
* If a comment is emotional with only a throwaway justification, it is reaction.
* If a comment is long but gives no verifiable evidence, it is hot_take (length is not evidence).

Comment:
{text}

Label:
```

***

## Notes for running it

* **Temperature 0** for determinism/reproducibility.
* **max_tokens** small (e.g. 4): we only need one word.
* The notebook flags unparseable responses. If >~10% are unparseable, the model is adding
  extra words; reinforce "Output ONLY the label" and re run. The `analysis|hot_take|reaction`
  normalization (lowercasing, `-`→`_`) is already handled in `prelabel.py` and should be
  mirrored in the notebook's parser if needed.
* **Record the baseline results** (overall accuracy + per class precision/recall/F1) before
  fine tuning. These go in the README evaluation report next to the fine tuned numbers.

## Hypothesis to test after fine tuning (planning.md §5)
The baseline will likely do *fine* on `reaction` (emotional language is obvious to a 70B
model) but struggle on the `analysis` vs `hot_take` boundary: the subjective "is this
evidence load bearing or decorative?" call. Note where it confuses those two; that's the gap
fine tuning needs to close.
