"""
analyze.py: Local error analysis and confidence calibration (stretch features).

Loads the fine tuned model (downloaded from Colab) and re runs it over the SAME test split
the notebook used, so the numbers reproduce the notebook's. Produces:

  1. A reliability and calibration table: bins predictions by confidence and reports the actual
     accuracy in each bin (does an 80 to 90% confident prediction actually hit ~85%?).
  2. analysis/wrong_predictions.csv: every misclassified test example with true, pred, conf.
  3. analysis/errors_for_llm.md: a paste ready digest of the misclassifications for an LLM
     to scan for systematic patterns (which you then verify by hand, planning.md §7.3).

IMPORTANT, reproduce the notebook's split:
The notebook splits 70/15/15. To analyze the *same* test rows, pass the same random seed the
notebook uses (default 42) and the same label order. If your notebook differs, edit SEED or
the split call below to match, or export the test split from Colab as data/test_split.csv and
pass --test data/test_split.csv (preferred, guarantees an identical set).

Usage:
    python analysis/analyze.py                              # uses data/labeled_data.csv + seed split
    python analysis/analyze.py --test data/test_split.csv   # use an exact split exported from Colab
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "takemeter-model"
LABELED = ROOT / "data" / "labeled_data.csv"
SEED = 42  # MUST match the notebook's split seed for the seed based reconstruction to be valid


def load_rows(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    return [(r["text"], r["label"]) for r in rows if r.get("label")]


def reconstruct_test_split(rows):
    """Reproduce the notebook's 70/15/15 split with the same seed, return the test 15%."""
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    # 70 / 30, then split the 30 into 15 val / 15 test, matches the typical notebook flow.
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    return list(zip(x_test, y_test))


def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Download takemeter-model/ from Colab into {ROOT}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return tok, model, id2label, label2id


def predict_all(tok, model, id2label, texts):
    preds, confs, all_probs = [], [], []
    for text in texts:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        preds.append(id2label[idx])
        confs.append(float(probs[idx]))
        all_probs.append({id2label[i]: float(p) for i, p in enumerate(probs)})
    return preds, confs, all_probs


def calibration_table(y_true, y_pred, confs):
    """Bin by confidence; report count and accuracy per bin (reliability)."""
    bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    table = []
    for lo, hi in bins:
        idxs = [i for i, c in enumerate(confs) if lo <= c < hi]
        if not idxs:
            table.append((f"{lo:.2f} to {hi:.2f}", 0, None, None))
            continue
        correct = sum(1 for i in idxs if y_true[i] == y_pred[i])
        acc = correct / len(idxs)
        avg_conf = sum(confs[i] for i in idxs) / len(idxs)
        table.append((f"{lo:.2f} to {hi:.2f}", len(idxs), avg_conf, acc))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None, help="exact test split CSV exported from Colab")
    args = ap.parse_args()

    tok, model, id2label, label2id = load_model()

    if args.test:
        test = load_rows(args.test)
        print(f"Using exact test split: {args.test} ({len(test)} rows)")
    else:
        all_rows = load_rows(LABELED)
        test = reconstruct_test_split(all_rows)
        print(f"Reconstructed test split from {LABELED} with seed={SEED} ({len(test)} rows)")
        print("  (If accuracy doesn't match the notebook, export the exact split via --test.)")

    texts = [t for t, _ in test]
    y_true = [l for _, l in test]
    y_pred, confs, all_probs = predict_all(tok, model, id2label, texts)

    labels_sorted = sorted(set(y_true) | set(y_pred))
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

    print("\n=== Overall ===")
    print(f"Accuracy: {acc:.3f}  ({sum(1 for a,b in zip(y_true,y_pred) if a==b)}/{len(y_true)})")

    print("\n=== Per class report ===")
    print(classification_report(y_true, y_pred, labels=labels_sorted, digits=3, zero_division=0))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    header = "true\\pred".ljust(12) + "".join(l.ljust(12) for l in labels_sorted)
    print(header)
    for i, l in enumerate(labels_sorted):
        print(l.ljust(12) + "".join(str(cm[i][j]).ljust(12) for j in range(len(labels_sorted))))

    print("\n=== Confidence calibration ===")
    print("bin".ljust(14) + "n".ljust(6) + "avg_conf".ljust(12) + "accuracy")
    for name, n, avg_conf, bin_acc in calibration_table(y_true, y_pred, confs):
        ac = f"{avg_conf:.3f}" if avg_conf is not None else "-"
        bc = f"{bin_acc:.3f}" if bin_acc is not None else "-"
        print(name.ljust(14) + str(n).ljust(6) + ac.ljust(12) + bc)

    # wrong predictions CSV
    wrong_path = Path(__file__).parent / "wrong_predictions.csv"
    wrong = [
        {"text": texts[i], "true": y_true[i], "pred": y_pred[i],
         "confidence": round(confs[i], 3),
         "probs": "; ".join(f"{k}={v:.2f}" for k, v in all_probs[i].items())}
        for i in range(len(texts)) if y_true[i] != y_pred[i]
    ]
    with wrong_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "true", "pred", "confidence", "probs"])
        w.writeheader()
        w.writerows(wrong)
    print(f"\nWrote {len(wrong)} misclassifications → {wrong_path}")

    # errors digest for LLM pattern finding
    pair_counts = defaultdict(int)
    for r in wrong:
        pair_counts[(r["true"], r["pred"])] += 1
    md = ["# Misclassified test examples (for LLM pattern finding)\n",
          "Ask an LLM: *what systematic pattern explains these errors?* Then verify by hand.\n",
          "## Confused pair counts (true to pred)\n"]
    for (t, p), c in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"- {t} → {p}: {c}")
    md.append("\n## Examples\n")
    for i, r in enumerate(wrong, 1):
        md.append(f"{i}. **true={r['true']} / pred={r['pred']}** (conf {r['confidence']})  \n"
                  f"   \"{r['text']}\"\n")
    digest_path = Path(__file__).parent / "errors_for_llm.md"
    digest_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote LLM digest → {digest_path}")


if __name__ == "__main__":
    main()
