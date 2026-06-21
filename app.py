"""
app.py: TakeMeter deployed interface (stretch feature).

A Gradio app: paste an r/nba comment, get the predicted discourse label
(analysis, hot_take, reaction) with a confidence score.

Loads the fine tuned DistilBERT you trained in Colab. Before running:
  1. In Colab, after training:  model.save_pretrained("takemeter-model");
     tokenizer.save_pretrained("takemeter-model")
  2. Download the takemeter-model/ folder and place it in this repo root
     (it's gitignored, too large to commit; document this in the README).
  3. pip install -r requirements.txt

Run:
    python app.py
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "takemeter-model"

# Visual identity per label (mirrors the lab3 app style).
LABEL_COLORS = {
    "analysis": "#16a34a",   # green: the "good take"
    "hot_take": "#ea580c",   # orange: bold but unsupported
    "reaction": "#2563eb",   # blue: in the moment feeling
    "unknown":  "#94a3b8",
}
LABEL_DESCRIPTIONS = {
    "analysis": "Structured argument backed by specific, verifiable evidence.",
    "hot_take": "A bold, confident opinion asserted without real supporting evidence.",
    "reaction": "An immediate emotional response to a play, game, or event.",
}

EXAMPLES = [
    "Their half court defensive rating is actually 3rd since the break, the regression is "
    "entirely in transition, where they fell from 12th to 27th. That's effort, not scheme.",
    "Jokic is the most skilled big to ever touch a basketball and it's not close. Anyone who "
    "disagrees doesn't watch basketball.",
    "NOOOO not again every single year man I can't do this 😭",
    "He's shooting 38% from three on high volume, that's completely fine, stop overreacting.",
    "The Lakers are making the Finals this year, book it. This roster is built for the playoffs.",
]


# =========================================================================== #
# Model loading (lazy, cached)
# =========================================================================== #
_model = None
_tokenizer = None
_id2label = None


def _load():
    global _model, _tokenizer, _id2label
    if _model is not None:
        return
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model folder not found at {MODEL_DIR}.\n"
            "Download takemeter-model/ from Colab (see app.py docstring or README)."
        )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()
    # id2label comes from the config saved during training; fall back to a sane default.
    cfg = _model.config.id2label
    _id2label = {int(k): v for k, v in cfg.items()} if cfg else {
        0: "analysis", 1: "hot_take", 2: "reaction"
    }


def predict(text: str):
    """Return (label, confidence, full_prob_dict)."""
    _load()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    label = _id2label.get(idx, "unknown")
    confidence = float(probs[idx])
    prob_dict = {_id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    return label, confidence, prob_dict


# =========================================================================== #
# UI rendering
# =========================================================================== #
def _result_html(label, confidence, prob_dict):
    color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
    desc = LABEL_DESCRIPTIONS.get(label, "")
    badge = (
        f"<span style='background:{color};color:white;padding:6px 14px;border-radius:999px;"
        f"font-weight:700;font-size:1.1em;'>{label}</span>"
        f"<span style='margin-left:12px;color:#374151;font-weight:600;'>"
        f"{confidence*100:.1f}% confident</span>"
    )
    # per class confidence bars
    bars = ""
    for lbl, p in sorted(prob_dict.items(), key=lambda kv: -kv[1]):
        c = LABEL_COLORS.get(lbl, "#94a3b8")
        bars += (
            f"<div style='margin:6px 0;'>"
            f"<div style='font-size:0.85em;color:#6b7280;'>{lbl}: {p*100:.1f}%</div>"
            f"<div style='background:#e5e7eb;border-radius:6px;overflow:hidden;height:10px;'>"
            f"<div style='width:{p*100:.1f}%;background:{c};height:10px;'></div></div></div>"
        )
    return f"""
<div style="font-family:sans-serif;padding:12px 0;">
  <div style="margin-bottom:14px;">{badge}</div>
  <p style="color:#6b7280;margin:0 0 14px 0;font-style:italic;">{desc}</p>
  {bars}
</div>
"""


def classify_comment(text: str) -> str:
    if not text.strip():
        return "<p style='color:#9ca3af;font-style:italic;'>Paste a comment above and click Classify.</p>"
    try:
        label, confidence, prob_dict = predict(text)
    except FileNotFoundError as e:
        return (
            "<div style='background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;"
            "border-radius:0 8px 8px 0;'><strong>⚠️ Model not loaded.</strong><br>"
            f"{str(e).splitlines()[0]}</div>"
        )
    return _result_html(label, confidence, prob_dict)


# =========================================================================== #
# Gradio app
# =========================================================================== #
THEME = gr.themes.Soft(primary_hue="green", secondary_hue="orange", neutral_hue="slate")

with gr.Blocks(title="TakeMeter", theme=THEME) as demo:
    gr.Markdown(
        """
# 🏀 TakeMeter
**Rate the discourse quality of an r/nba comment.**

Classifies a comment as **analysis** (evidence backed argument), **hot_take**
(confident opinion, no evidence), or **reaction** (in the moment feeling), with a
confidence score from a fine tuned DistilBERT.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            comment_box = gr.Textbox(
                label="r/nba comment",
                placeholder="Paste a comment here…",
                lines=6,
            )
            classify_btn = gr.Button("Classify →", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("#### Result")
            result_html = gr.HTML(
                value="<p style='color:#9ca3af;font-style:italic;'>Result will appear here.</p>"
            )

    gr.Markdown("#### Try an example")
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=comment_box)

    classify_btn.click(classify_comment, inputs=comment_box, outputs=result_html)
    comment_box.submit(classify_comment, inputs=comment_box, outputs=result_html)


if __name__ == "__main__":
    demo.launch()
