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

### Label distribution
| Label | Count | % |
|---|---|---|
| analysis | 83 | 28.2% |
| hot_take | 82 | 27.9% |
| reaction | 129 | 43.9% |
| **Total** | **294** | 100% |

Every class clears the 20% floor and sits well under the 70% ceiling. reaction is the largest,
which is expected for r/nba (game thread emotion is the most common register), but not dominant.
The set is 300 collected comments minus 6 automoderator and mod removal messages that were
dropped as non discourse.

### Three genuinely difficult examples

1. **Long but evidence free rant (id `osyvx7q`).** *"I've watched Giannis play for years and I
   don't have memory rot... 2 weeks of good play is nothing... people always remember the
   highlights but not the games he handed to us by bricking jump shot after jump shot."* The
   length and paragraph structure pattern match to `analysis`, but the comment offers only
   assertion and memory, no verifiable evidence. Labeled `hot_take` on the rule *length is not
   evidence*.
2. **One load bearing stat (id `oswc8z5`).** *"Castle is 21 and was 6th in assists lol."* Very
   short and casual, but the single stat (6th in assists) is doing the real argumentative work
   in a debate about whether Castle is a passer, not decorating an opinion. Labeled `analysis`.
   Another commenter (id `oswadsy`) replied to correct it to "9th in the league," which confirms
   the stat was a genuine, checkable claim rather than a flourish.
3. **Emotional venting that contains a take (id `osytrru`).** *"Ah now he's trying to compensate
   and rewrite the narrative around his ring... Dude just stop.....you went to a super team to
   get a ring. Stop the bullshit."* It embeds an opinion (joined a super team for a ring) inside
   what is dominantly emotional venting at a person. Labeled by *dominant function*: the register
   is emotional and in the moment, so `reaction`, even though a take is buried in it.

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
*Test set n = 45 (15% of 294).*

| Metric | Zero shot baseline | Fine tuned DistilBERT |
|---|---|---|
| Overall accuracy | ⏳ pending (Section 5) | 0.622 |
| Macro F1 | ⏳ pending (Section 5) | 0.55 |

### 6.2 Per class metrics
| Label | Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| analysis | baseline | ⏳ | ⏳ | ⏳ | 13 |
| analysis | fine tuned | 0.80 | 0.31 | 0.44 | 13 |
| hot_take | baseline | ⏳ | ⏳ | ⏳ | 12 |
| hot_take | fine tuned | 0.42 | 0.42 | 0.42 | 12 |
| reaction | baseline | ⏳ | ⏳ | ⏳ | 20 |
| reaction | fine tuned | 0.68 | 0.95 | 0.79 | 20 |

### 6.3 Confusion matrix (fine tuned, test set)
*Rows = true label, columns = predicted. Also committed as
[`outputs/confusion_matrix.png`](outputs/confusion_matrix.png).*

| true \ pred | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | 4 | 6 | 3 |
| **hot_take** | 1 | 5 | 6 |
| **reaction** | 0 | 1 | 19 |

*Reading: the diagonal (4, 5, 19) is correct. The two heavy off diagonal cells are
analysis to hot_take (6) and hot_take to reaction (6), the two subjective boundaries. reaction
is almost never missed (19 of 20).*

### 6.4 Three wrong predictions, analyzed

Context: all 17 of the fine tuned model's test errors came with confidence between 0.34 and
0.37, barely above the 0.33 random floor. The model is not confidently wrong, it is uncertain,
and that uncertainty falls on exactly the two subjective boundaries. The three below are chosen
to cover each confusion direction (analysis to hot_take, hot_take to reaction, analysis to
reaction).

1. **Comment (id `ost0weg`):** *"I didnt read the article but... 11.6k in expenses (140k a
   year) 7k baby mama fees... even if you cut 60% of his nba earnings for taxes and agent fees
   thats 70 million left over... If he just saved 40 million... thats 400k for 100 years after
   taxes... His current lifestyle is more than manageable."* **true** `analysis` / **pred**
   `hot_take` (conf 0.34). *Why:* this is the core `analysis` to `hot_take` failure. The comment
   is a multi step quantitative argument (it does real arithmetic to reach a conclusion), but
   the model reads it as just another confident opinion. It has not learned that the numbers
   here are load bearing rather than decorative. This is a **data and task** problem, not a
   labeling one: with only about 206 training examples, the model never saw enough worked
   numeric arguments to separate "math that supports a claim" from "a stat dropped for effect."
   Fix: more `analysis` examples that reason through numbers, and more `hot_take` examples that
   cite a stat decoratively, so the boundary is drawn by *how* the number is used.

2. **Comment (id `osyv4hk`):** *"I'd prefer we spend the next season with Maluach utilized for
   20+ minutes to develop."* **true** `hot_take` / **pred** `reaction` (conf 0.37). *Why:* the
   `hot_take` to `reaction` failure. This is a calm, standalone roster opinion with no evidence,
   which is the definition of a hot take, but it is short and casually worded, so the model
   keys on register and files it as an in the moment reaction. The model appears to use tone and
   length as a proxy for `reaction` rather than detecting whether a standalone claim is being
   made. Fix: more short, calm `hot_take` examples so brevity and a low key tone stop being
   read as emotional.

3. **Comment (id `osyua9k`):** *"It was literally the same script everytime, Luka dropping
   carrying and dropping 30/10/10 thru 3 quarters then wearing down while Kawhi just didnt miss
   in the 4th."* **true** `analysis` / **pred** `reaction` (conf 0.34). *Why:* this is an
   observed tactical pattern backed by a stat line (30/10/10, the fourth quarter fade), which is
   `analysis` by the rubric, yet it was predicted `reaction`. It shows that **numbers alone do
   not trigger `analysis`** in this model: combined with the 0.31 `analysis` recall, the model
   rarely commits to `analysis` at all and defaults to the easier classes. It is also one of the
   genuinely contestable one stat calls from annotation, so the model's confusion mirrors the
   human difficulty at this boundary rather than being plainly wrong.

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
