# TakeMeter: planning.md

*AI201 Project 3. Design document written **before** data collection and updated as
annotation surfaced new edge cases. This is my working notebook; the README is the polished
final report.*

***

## 1. Community

**Choice: r/nba** (the main NBA subreddit, ~15M members).

**Why this community.** r/nba is a high volume, text heavy community where "what makes a
good take" is a *native* concern: users routinely accuse each other of posting "hot takes,"
praise "actual analysis," and dismiss "just reaction." The phrase "hot take" is literally
community vocabulary, which means my labels map onto a distinction people there already make.

**Why it's a good fit for a classification task.** Discourse quality varies enormously
within a single thread: a game thread mixes one word emotional reactions ("WHAT A SHOT"),
confident unsupported opinions ("Luka is already top 5 all time"), and genuinely reasoned
breakdowns (lineup data, playoff splits, tactical observations). That variance is what makes
the task nontrivial: the same *topic* (e.g. "is Player X good?") appears across all three
labels, so the model can't just key off topic words; it has to pick up on *structure and
evidence*. That's exactly the kind of distinction fine tuning is supposed to learn.

***

## 2. Labels

Three mutually exclusive labels. The decision axis is **how the claim is supported**, not
*what* the claim is about or whether I agree with it.

### `analysis`
> A comment that makes a structured argument backed by **specific, verifiable evidence**:
> statistics, historical comparison, tactical/film observation, or a clear causal chain.
> If you removed the opinion framing, a real argument would remain.

**Examples:**
1. *"People say the Wolves' defense fell off, but their half court defensive rating is
   actually 3rd since the all star break. The regression is entirely in transition, where
   they went from 12th to 27th. That's a effort/personnel problem, not a scheme problem."*
2. *"Comparing Wemby's rookie year to KAT's: Wemby's already a better rim protector (3.6
   BPG vs 1.7) while shooting a similar volume from three. The defensive gap is the whole
   case for him being a different tier of prospect."*

### `hot_take`
> A **bold, confident opinion asserted without genuine supporting evidence.** The claim may
> be right, but the comment *asserts* rather than *argues*. Decorative or cherry picked
> stats that exist to sound credible (rather than to reason) still count as hot_take.

**Examples:**
1. *"Jokic is the most skilled big man to ever touch a basketball and it's not close.
   Anyone who disagrees doesn't watch basketball."*
2. *"The Lakers are making the Finals this year, book it. This roster is built for the
   playoffs."*

### `reaction`
> An **immediate emotional response** to a specific play, game, or event. Little to no
> argument: the comment is expressing a feeling in the moment (hype, despair, shock, humor).

**Examples:**
1. *"NOOOO not again 😭 every single year man I can't do this"*
2. *"BANG. game over. I'm levitating right now"*

### Why these three matter to the community
They reflect the exact ladder r/nba users themselves apply to discourse: *reaction* (fine in
a game thread, low effort in a discussion thread), *hot_take* (engagement bait, often
downvoted in serious threads), and *analysis* (the thing that gets gilded). A tool that sorts
comments along this axis is recognizably measuring "take quality" the way a regular would.

***

## 3. Hard edge cases (decision rules)

Designed *before* annotation; expanded with real cases found *during* annotation (§3.4+).

### 3.1 The one stat post: `analysis` vs `hot_take`
*"LeBron is overrated, his playoff record vs top seeded opponents is below .500."*
* **Rule:** If the evidence would support the claim *with the opinion framing removed*, and
  the stat is doing real reasoning work, then `analysis`. If the stat is decorative or
  cherry picked to *sound* credible while the comment is really just asserting, then `hot_take`.
* **This example is `hot_take`:** the framing is accusatory ("overrated"), and a single
  context stripped win rate stat is selected for effect, not as part of an argument.

### 3.2 Reaction with a reason: `reaction` vs `analysis`/`hot_take`
*"That's why he's the MVP, unreal closing ability."*
* **Rule:** A short emotional burst with a *throwaway* justification is still `reaction`.
  It only escalates to `hot_take` if the *claim* is the point (a standalone opinion), or to
  `analysis` if real evidence is given. Emotional register plus tied to a live moment: keep it
  `reaction`.

### 3.3 Sarcasm / jokes
* **Rule:** Label by communicative *function*. A sarcastic one liner reacting to a play is
  `reaction`. A joke that's really smuggling a confident opinion is `hot_take`. Sarcasm is the
  single biggest anticipated source of model error (the surface words invert the meaning).

### 3.4 *(filled during annotation: 3 required hard cases)*
* **Case A, long but evidence free rant.** *e.g. a 6 sentence paragraph arguing the refs
  are rigging games against a team, with zero specifics.* Length pattern matches to
  `analysis`, but there's no verifiable evidence, so **`hot_take`**. Decision: *length is not
  evidence.* (TODO: confirm the exact comment + id during labeling.)
* **Case B, quoted stat with a real but thin argument.** *e.g. "He's shooting 38% from
  three, that's fine, stop overreacting."* One real stat plus a mild claim, so **`analysis`**
  (the stat is load bearing, not decorative), even though it's short. (TODO: confirm comment.)
* **Case C, multi sentence emotional venting after a loss that drifts into a take.**
  Starts as `reaction`, ends with "we need to fire the GM." Decision rule: label by the
  *dominant* function / final claim. If it lands on a standalone opinion, then `hot_take`;
  if it stays venting, then `reaction`. (TODO: confirm comment + which way I called it.)

> These three TODOs get replaced with the actual comments + ids while labeling, then copied
> into the README's "difficult to label examples" section.

***

## 4. Data collection plan

* **Source:** public comments on r/nba, pulled with `collect.py`. Reddit blocks
  unauthenticated requests (403) from this network, and the official API host was also
  unreachable, so the default backend is the **arctic-shift public reddit archive**, a no auth
  mirror of public reddit data. PRAW and the direct `.json` endpoints remain as alternate
  backends (`--source`) for any network where reddit is reachable. Public content only, no
  auth walled or private data.
* **Source mix (to guarantee label variety):** ~65% from **r/nba** (a broad discourse mix:
  game thread reactions, confident `hot_take` opinions, and some breakdowns) and ~35% from
  **r/nbadiscussion** (rich in `analysis`). Pulling from two sources prevents the model from
  keying on one subreddit's style, and the real labels are assigned per comment by content,
  not by source.
* **Volume:** collect ~300 raw comments, then label down to **≥200** clean examples.
* **Target distribution:** aim **~33% each**, hard floor **≥20% per class**, hard ceiling
  **≤70%** for any class (assignment requirement).
* **Filters in collect.py:** drop `[deleted]`/`[removed]`, AutoModerator/bot comments, pure
  links/emoji only, and comments under ~4 words (too short to label meaningfully, though a
  short emotional burst CAN be a valid `reaction`, so the floor is low). Dedupe on text.
* **If a label is underrepresented after 200:** targeted second pass, `reaction` from more
  Game Threads, `analysis` from r/nbadiscussion style serious threads, `hot_take` from
  "unpopular opinion"/"hot take" megathreads, until each class clears 20%.

***

## 5. Evaluation metrics

Accuracy alone is insufficient because the classes may end up mildly imbalanced and the
*errors are not equally interesting*. Confusing `analysis` with `hot_take` (the subtle boundary)
matters more than `reaction` being easy. So:

* **Overall accuracy:** headline number, and the fairest single comparison vs the baseline.
* **Per class precision / recall / F1:** to see *which* distinction the model actually
  learned. I specifically expect the `analysis` vs `hot_take` boundary to be the weak one;
  per class F1 is the only way to surface that.
* **Macro F1:** the primary single metric. It weights each class equally, so a model that
  nails the easy `reaction` class but can't tell `analysis` from `hot_take` is correctly
  penalized (plain accuracy would hide it).
* **Confusion matrix:** to read the *direction* of errors (e.g. analysis to hot_take vs
  hot_take to analysis), which is the actionable signal for what to fix.
* **(Stretch) confidence calibration:** whether softmax confidence is trustworthy.
* **(Stretch) Cohen's kappa:** inter annotator agreement, to establish a human ceiling: if
  two humans only agree ~75%, I shouldn't expect the model to "beat" that.

***

## 6. Definition of success

Concrete, checkable thresholds (so I can objectively grade myself at the end):

* **Minimum bar (fine tuning "worked"):** fine tuned **macro F1 ≥ 0.65** AND it beats the
  zero shot Groq baseline on accuracy by **≥ 10 percentage points**. Random guessing on 3
  classes is 0.33, so this must clearly exceed that.
* **Good enough for deployment in a real community tool:** **macro F1 ≥ 0.75** with **no
  single class F1 < 0.60** (every distinction usable), AND calibration good enough that a
  ">80% confidence" prediction is right ≥80% of the time, so a real tool could show only
  high confidence labels and abstain otherwise.
* **Honest expectation:** the `analysis`/`hot_take` boundary is genuinely subjective; if
  human inter annotator kappa is ~0.6 to 0.7, a model at macro F1 ~0.70 is roughly at the human
  ceiling and I'll say so rather than chasing a misleading higher number.

***

## 7. AI Tool Plan

This project has little code to generate, so AI tools help at three specific points:

1. **Label stress testing (before annotating).** Give Claude the §2 definitions + §3 edge
   cases and ask it to generate 5 to 10 comments that sit *on the boundary* between two labels.
   If I can't cleanly classify what it produces, my definitions are too loose, so tighten them
   before labeling 200. *Outcome + any definition changes recorded in the README AI section.*
2. **Annotation assistance (optional, disclosed).** Use Groq (`prelabel.py`) to prelabel the
   raw comments against the §2 definitions, then **review and correct every label** by hand.
   Tracking: the `notes` column flags rows that were prelabeled and changed, so I can
   report how often the LLM disagreed with my final call. Disclosed in the README.
3. **Failure analysis (after fine tuning).** Paste the misclassified test examples
   (`analysis/errors_for_llm.md`) into Claude and ask it to find *systematic* patterns
   (sarcasm, short posts, one specific confused label pair). Then **verify each claimed
   pattern by rereading the examples myself**: only verified patterns go in the report,
   and I note anything the LLM claimed that I had to discard.

***

## 8. Stretch features (all four planned)

> Per the assignment, this section is updated before starting each stretch feature.

1. **Deployed interface** (`app.py`): Gradio UI: paste a comment, get predicted label +
   confidence, modeled on my lab3 podclassifier app.
2. **Confidence calibration** (`analysis/analyze.py`): bin test predictions by confidence
   and check whether high confidence predictions are actually more accurate (reliability table).
3. **Error pattern analysis** (`analysis/analyze.py` + step 7.3): go beyond listing wrong
   predictions to name a *systematic* failure mode (hypothesis: sarcasm and analysis vs hot_take).
4. **Inter annotator reliability** (`analysis/iaa.py`, `data/iaa_subset.csv`): a second
   person independently labels 30+ examples; report Cohen's kappa + % agreement and analyze
   where/why we disagreed. *Human dependency: requires a second annotator.*

***

## 9. Workflow checklist

* [ ] Read 30 to 40 real r/nba comments; confirm/refine the §2 labels.
* [ ] `collect.py` to `data/raw_comments.csv` (~300, thread mixed).
* [ ] (opt) `prelabel.py`, review every label, then `data/labeled_data.csv` (≥200).
* [ ] Confirm distribution (each class ≥20%, none >70%); fill §3.4 with real cases.
* [ ] Upload CSV to Colab; run Sections 1 to 2; write baseline prompt; run Section 5 (baseline).
* [ ] Run Sections 3 to 4 (fine tune + eval); note hyperparameter decision; Section 6 (compare).
* [ ] Download `evaluation_results.json`, `confusion_matrix.png`, and `takemeter-model/`.
* [ ] Stretch: `app.py`, `analyze.py` (calibration + errors), `iaa.py` (2nd annotator).
* [ ] Fill README results sections; record 3 to 5 min demo; push to GitHub.
