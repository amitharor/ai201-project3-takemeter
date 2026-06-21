"""
iaa.py: Inter annotator reliability (stretch feature).

Two modes:

  EXPORT: pull 30 examples from your labeled data, strip the labels, and write a blank
  subset for a SECOND person to label independently:
      python analysis/iaa.py --export --n 30

  This writes data/iaa_subset.csv with columns: id, text, label  (label left blank).
  Give that file to your second annotator. They fill the `label` column with
  analysis, hot_take, or reaction, save it, and send it back as data/iaa_subset_annotator2.csv.

  COMPARE: once you have both sets of labels, compute agreement:
      python analysis/iaa.py --compare

  This matches rows by `id`, prints percent agreement and Cohen's kappa, and lists every
  disagreement so you can analyze where the two of you diverged (planning.md §8.4).

Cohen's kappa is computed from scratch (no extra deps) so this runs even without sklearn.
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "labeled_data.csv"
SUBSET = ROOT / "data" / "iaa_subset.csv"
SUBSET_A2 = ROOT / "data" / "iaa_subset_annotator2.csv"
LABELS = ["analysis", "hot_take", "reaction"]


def export(n):
    rows = list(csv.DictReader(LABELED.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {LABELED}. Label your data first.")
    # Take a roughly stratified, evenly spaced sample so all classes appear in the subset.
    rows_sorted = sorted(rows, key=lambda r: r.get("label", ""))
    step = max(len(rows_sorted) // n, 1)
    sample = rows_sorted[::step][:n]
    # ensure each row has a stable id
    with SUBSET.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        w.writeheader()
        for i, r in enumerate(sample):
            w.writerow({"id": r.get("id") or f"row{i}", "text": r["text"], "label": ""})
    print(f"Wrote {len(sample)} blank examples → {SUBSET}")
    print("Give this to your second annotator. They fill `label` and return it as")
    print(f"  {SUBSET_A2.name}  (same id + text, label completed).")


def _load_labels(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    return {r["id"]: r["label"].strip() for r in rows if r.get("label", "").strip()}


def cohens_kappa(pairs):
    """pairs: list of (label_a, label_b). Returns kappa."""
    n = len(pairs)
    if n == 0:
        return None
    # observed agreement
    po = sum(1 for a, b in pairs if a == b) / n
    # expected agreement from marginals
    from collections import Counter
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(ca) | set(cb))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def compare():
    # My labels for the subset come from the main labeled file, matched by id.
    mine_all = _load_labels(LABELED)
    if not SUBSET_A2.exists():
        raise SystemExit(f"Annotator 2 file not found: {SUBSET_A2}. (See --export.)")
    theirs = _load_labels(SUBSET_A2)

    common = [i for i in theirs if i in mine_all]
    if not common:
        raise SystemExit("No overlapping ids between your labels and annotator 2. "
                         "Make sure the `id` column was preserved.")

    pairs = [(mine_all[i], theirs[i]) for i in common]
    agree = sum(1 for a, b in pairs if a == b)
    pct = agree / len(pairs)
    kappa = cohens_kappa(pairs)

    print(f"Compared {len(pairs)} examples labeled by both annotators.\n")
    print(f"Percent agreement: {pct*100:.1f}%  ({agree}/{len(pairs)})")
    print(f"Cohen's kappa:     {kappa:.3f}")
    print(_kappa_reading(kappa))

    disagreements = [(i, mine_all[i], theirs[i]) for i in common if mine_all[i] != theirs[i]]
    if disagreements:
        print(f"\n=== Disagreements ({len(disagreements)}) ===")
        text_by_id = {r["id"]: r["text"] for r in csv.DictReader(SUBSET.open(encoding="utf-8"))} \
            if SUBSET.exists() else {}
        for i, m, t in disagreements:
            snippet = (text_by_id.get(i, "")[:100] + "…") if text_by_id.get(i) else ""
            print(f"- id={i}: you={m} / them={t}   {snippet}")


def _kappa_reading(k):
    if k is None:
        return ""
    if k < 0.2:   band = "slight"
    elif k < 0.4: band = "fair"
    elif k < 0.6: band = "moderate"
    elif k < 0.8: band = "substantial"
    else:         band = "almost perfect"
    return f"(Landis & Koch: {band} agreement)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write a blank subset for annotator 2")
    ap.add_argument("--compare", action="store_true", help="compute kappa from both label sets")
    ap.add_argument("--n", type=int, default=30, help="subset size for --export")
    args = ap.parse_args()

    if args.export:
        export(args.n)
    elif args.compare:
        compare()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
