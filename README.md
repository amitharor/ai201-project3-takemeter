# 🏀 TakeMeter: r/nba Discourse Quality Classifier

A fine tuned text classifier that rates the quality of a take in r/nba discussion. Given a
comment, it predicts whether the comment is **`analysis`** (an evidence backed argument),
**`hot_take`** (a confident opinion with no real support), or **`reaction`** (an in the moment
emotional response).

*AI201 Project 3. Design notes live in [`planning.md`](planning.md); this README is the final
report.*

> **Status:** scaffolding + design complete. Sections marked `⏳ FILL AFTER COLAB` are
> populated after fine tuning in the course Colab notebook.

***

## 1. Community choice and reasoning

I chose **r/nba**. "What makes a good take" is *native* community vocabulary there. Users
routinely call each other out for "hot takes," praise "actual analysis," and dismiss "just
reaction." A single game thread mixes one word emotional bursts, confident unsupported
opinions, and genuinely reasoned breakdowns, often about the *same* topic. That topic overlap
is what makes the task nontrivial: the classifier can't key off subject matter, it has to
learn *how a claim is supported*. See [`planning.md §1`](planning.md).

## 2. Label taxonomy

The decision axis is **how the claim is supported**, not the topic or whether the claim is
correct. Labels are mutually exclusive.

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| **analysis** | Structured argument backed by specific, verifiable evidence (stats, historical comparison, tactical observation). Strip the opinion framing and a real argument remains. | *"Their half court D rating is 3rd since the break. The regression is entirely in transition (12th, 27th). That's effort, not scheme."* | *"Wemby's already a better rim protector than rookie KAT (3.6 vs 1.7 BPG) on similar 3pt volume. That's the whole case for him being a different tier."* |
| **hot_take** | A bold, confident opinion asserted without genuine evidence. Decorative, cherry picked stats that only sound credible still count. | *"Jokic is the most skilled big to ever touch a basketball and it's not close. Anyone who disagrees doesn't watch basketball."* | *"The Lakers are making the Finals this year, book it."* |
| **reaction** | Immediate emotional response to a play/game/event. Little to no argument. | *"NOOOO not again every single year man I can't do this 😭"* | *"BANG. game over. I'm levitating right now."* |

## 3. Dataset

* **Source:** public r/nba comments collected with [`collect.py`](collect.py) (PRAW, with a
  no auth `.json` fallback), sampled across a **mix of thread types** (game and post game threads
  that are reaction rich, r/nbadiscussion serious threads that are analysis rich, and opinion bait posts
  that are hot_take rich) so labels don't correlate with thread type. Public content only.
* **Labeling process:** each comment read and labeled by hand against the
  [`planning.md §2`](planning.md) definitions. *(Optionally prelabeled by Groq via
  [`prelabel.py`](prelabel.py) and then reviewed and corrected row by row: see §11 AI usage.)*
* **File:** [`data/labeled_data.csv`](data/labeled_data.csv) (`text, label, notes`), a single
  file; the notebook does the 70/15/15 split.

### Label distribution ⏳ FILL AFTER LABELING
| Label | Count | % |
|---|---|---|
| analysis | _ | _ |
| hot_take | _ | _ |
| reaction | _ | _ |
| **Total** | **_** | 100% |

*(Target: each class ≥20%, none >70%.)*

### Three genuinely difficult examples ⏳ FILL AFTER LABELING
*(Copy the resolved versions of `planning.md §3.4` cases A/B/C here, with the actual comment
text and the call I made plus why.)*

1. **Long but evidence free rant** → labeled `hot_take`. *Why:* length pattern matches to
   analysis, but there was no verifiable evidence: *length is not evidence.*
2. **Thin one stat argument** → labeled `analysis`. *Why:* the single stat was load bearing,
   not decorative, even though the comment was short.
3. **Emotional venting that drifts into a take** → labeled by *dominant function* or final
   claim. *Why:* … *(record the actual call)*.

## 4. Fine tuning approach

* **Base model:** `distilbert-base-uncased` (HuggingFace), a small, fast encoder well suited
  to a 3 class sequence classification task on a few hundred examples.
* **Setup:** course Colab notebook, free **T4 GPU**, `transformers` Trainer. 70/15/15
  train/val/test split (stratified).
* **Key hyperparameter decision:** ⏳ *(start from the notebook defaults, 3 epochs, lr 2e-5,
  batch 16, and document the one change you make. Planned decision: bump epochs to ~5 if the
  validation loss is still falling at epoch 3, since 200 examples is small and underfits
  quickly. Record what you actually chose and why.)*

## 5. Baseline (zero shot Groq)

* **Model:** `llama-3.3-70b-versatile`, temperature 0, classifying each **test** comment with
  no task specific training.
* **Prompt:** the exact prompt in [`baseline_prompt.md`](baseline_prompt.md) (label
  definitions verbatim from planning.md plus "output only the label name").
* **How results were collected:** run in the notebook's Section 5 over the locked test split;
  unparseable responses are flagged and the prompt tightened if >~10% fail to parse.

***

## 6. Evaluation report ⏳ FILL AFTER COLAB

### 6.1 Headline metrics
| Metric | Zero shot baseline | Fine tuned DistilBERT |
|---|---|---|
| Overall accuracy | _ | _ |
| Macro F1 | _ | _ |

### 6.2 Per class metrics
| Label | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| analysis | baseline | _ | _ | _ |
| analysis | fine tuned | _ | _ | _ |
| hot_take | baseline | _ | _ | _ |
| hot_take | fine tuned | _ | _ | _ |
| reaction | baseline | _ | _ | _ |
| reaction | fine tuned | _ | _ | _ |

### 6.3 Confusion matrix (fine tuned, test set)
*Rows = true label, columns = predicted. Also committed as
[`outputs/confusion_matrix.png`](outputs/confusion_matrix.png).*

| true \ pred | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | _ | _ | _ |
| **hot_take** | _ | _ | _ |
| **reaction** | _ | _ | _ |

### 6.4 Three wrong predictions, analyzed ⏳ FILL AFTER COLAB
*(Pull from `analysis/wrong_predictions.csv`. For each: the comment, true vs predicted, and a
real diagnosis using the guiding questions: which boundary failed, why it's hard, whether
it's a labeling vs data problem, and what would fix it.)*

1. **Comment:** … **true** `analysis` / **pred** `hot_take`. *Why:* …
2. **Comment:** … **true** `_` / **pred** `_`. *Why:* …
3. **Comment:** … **true** `_` / **pred** `_`. *Why:* …

### 6.5 Sample classifications (fine tuned)
*3 to 5 posts run through the model with predicted label + confidence. At least one correct
example explained.*

| Comment | Predicted | Confidence | Note |
|---|---|---|---|
| _ | _ | _% | ✅ correct, *why this is reasonable: …* |
| _ | _ | _% | |
| _ | _ | _% | |

***

## 7. Reflection: what the model learned vs. what I intended ⏳ FILL AFTER COLAB

*(Higher level than the error list. Did the model learn "evidence vs assertion," or a proxy
like length, presence of numbers, or capslock? What did it overfit to and what did it miss?
e.g. "I intended it to judge whether evidence is load bearing; in practice it appears to treat
any numeric token as a signal for `analysis`, which is why cherry picked stat hot_takes fool
it.")*

## 8. Spec reflection

* **One way the spec (`planning.md`) helped:** ⏳ *(e.g. forcing the one stat decision rule in
  §3.1 before annotating kept analysis vs hot_take labeling consistent across 200 examples.)*
* **One way the implementation diverged from the spec, and why:** ⏳ *(record an actual
  divergence, e.g. a label boundary you had to redraw after seeing real data, or a class you
  had to overcollect to clear the 20% floor.)*

## 9. Stretch features

* **Deployed interface:** [`app.py`](app.py): Gradio UI, paste a comment → label + confidence
  bars. See §10 to run.
* **Confidence calibration:** [`analysis/analyze.py`](analysis/analyze.py): reliability table
  binning predictions by confidence vs actual accuracy. ⏳ *findings here.*
* **Error pattern analysis:** beyond individual errors, the systematic pattern (hypothesis:
  sarcasm + the analysis↔hot_take boundary). ⏳ *verified pattern here.*
* **Inter annotator reliability:** [`analysis/iaa.py`](analysis/iaa.py): a second annotator
  labeled 30 examples; Cohen's kappa + agreement + disagreement analysis. ⏳ *numbers here.*

## 10. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add REDDIT_* (optional) and GROQ_API_KEY

# 1. Collect raw comments
python collect.py --target 300
# 2. (optional) Groq prelabel, then review every row by hand into data/labeled_data.csv
python prelabel.py
# 3. Fine tune + baseline in the Colab notebook (upload data/labeled_data.csv).
#    Download evaluation_results.json + confusion_matrix.png → outputs/,
#    and save_pretrained → download takemeter-model/ into the repo root.

# 4. Deployed interface (needs takemeter-model/)
python app.py
# 5. Calibration + error analysis (needs takemeter-model/)
python analysis/analyze.py
# 6. Inter annotator reliability
python analysis/iaa.py --export --n 30     # give data/iaa_subset.csv to a 2nd person
python analysis/iaa.py --compare           # after they return iaa_subset_annotator2.csv
```

> **Note:** `takemeter-model/` is gitignored (too large to commit). Download it from Colab
> via `save_pretrained` before running `app.py` / `analyze.py`.

## 11. AI usage

*(At least 2 specific instances: what I directed the tool to do, what it produced, what I
changed or overrode. Disclose any annotation assistance.)*

1. **Label stress testing.** I gave Claude my draft label definitions + edge cases and asked
   it to generate boundary case comments. ⏳ *(record which definitions I tightened as a result.)*
2. **Failure pattern analysis.** I pasted `analysis/errors_for_llm.md` into an LLM and asked
   for systematic error patterns, then verified each by rereading the examples. ⏳ *(record
   what it found and what I discarded.)*
3. **(If used) Annotation assistance.** Groq prelabeled raw comments via `prelabel.py`; I
   reviewed and corrected **every** label by hand (changes flagged in the `notes` column). ⏳
   *(record how often the LLM disagreed with my final call.)*

## 12. Demo video

⏳ *(link to the 3 to 5 min demo: 3 to 5 comments classified with label + confidence, one correct
narrated, one incorrect narrated, and a walkthrough of the evaluation report.)*
