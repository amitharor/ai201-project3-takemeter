"""
prelabel.py: OPTIONAL: use Groq to prelabel raw comments, then YOU review every row.

This is an annotation *accelerator*, not a replacement. It writes a `pred_label` column
(the LLM's guess) and a `label` column initialized to the same value. You then open the CSV
and CORRECT every label by hand. The `notes` column is where you flag what you changed.
Disclosed in the README's AI usage section (see planning.md §7.2).

Usage:
    python prelabel.py                         # reads data/raw_comments.csv
    python prelabel.py --in data/raw_comments.csv --out data/prelabeled.csv

Requires GROQ_API_KEY in .env. Uses the same label definitions as the zero shot baseline
(baseline_prompt.md) so the prelabels are consistent with how the baseline is judged.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
VALID = {"analysis", "hot_take", "reaction"}

# Keep this prompt aligned with baseline_prompt.md and planning.md §2.
SYSTEM_PROMPT = """You classify r/nba comments by how a claim is supported. Output EXACTLY one label, nothing else.

Labels:
- analysis: a structured argument backed by specific, verifiable evidence (stats, historical comparison, tactical/film observation, clear causal chain). Removing the opinion framing still leaves a real argument.
- hot_take: a bold, confident opinion asserted WITHOUT genuine supporting evidence. Decorative or cherry picked stats that only exist to sound credible still count as hot_take.
- reaction: an immediate emotional response to a play/game/event. Little to no argument; expressing a feeling in the moment.

Respond with only one word: analysis, hot_take, or reaction."""


def classify(client, text):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    raw = resp.choices[0].message.content.strip().lower()
    # normalize common variants
    raw = raw.replace("-", "_").replace(" ", "_")
    for label in VALID:
        if label in raw:
            return label
    return "UNPARSEABLE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw_comments.csv")
    ap.add_argument("--out", dest="out", default="data/prelabeled.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set in .env"); sys.exit(1)

    in_path = Path(args.inp)
    if not in_path.exists():
        print(f"Input not found: {in_path}. Run collect.py first."); sys.exit(1)

    rows = list(csv.DictReader(in_path.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    counts = {}
    for i, row in enumerate(rows, 1):
        label = classify(client, row["text"])
        row["pred_label"] = label
        row["label"] = label if label in VALID else ""   # you will correct this
        row["notes"] = ""
        counts[label] = counts.get(label, 0) + 1
        if i % 20 == 0:
            print(f"  {i}/{len(rows)} labeled...")
        time.sleep(0.2)  # stay under free tier rate limits

    out_path = Path(args.out)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} prelabeled rows to {out_path}")
    print("Prelabel distribution:", counts)
    print("\n*** REVIEW EVERY ROW. *** Correct the `label` column by hand, note any change")
    print("in `notes`, then save the reviewed file as data/labeled_data.csv.")


if __name__ == "__main__":
    main()
