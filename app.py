"""
Narrative Coreference Resolver - Gradio demo (transformer version)

Install (use Python 3.10 or 3.11, not 3.14):
    pip install fastcoref gradio

Run:
    python app.py

MODEL_NAME:
    "fcoref"   -> F-Coref (DistilRoBERTa): fast, fine on CPU
    "lingmess" -> LingMess (Longformer-large): most accurate, bigger download, slower on CPU
"""
import torch
import gradio as gr

MODEL_NAME = "fcoref"      # "fcoref" or "lingmess"
DEVICE = "cpu"             # use "cuda:0" if you have an NVIDIA GPU with CUDA PyTorch

print(f"Loading {MODEL_NAME} (the first run downloads the model)...")
import transformers
transformers.PreTrainedModel.all_tied_weights_keys = {}

if MODEL_NAME == "lingmess":
    from fastcoref import LingMessCoref
    model = LingMessCoref(device=DEVICE)
else:
    from fastcoref import FCoref
    model = FCoref(device=DEVICE)
print("Model ready.")


def resolve_coref(text):
    """Returns (highlighted_text_pieces, cluster_summary_text)."""
    if not text or not text.strip():
        return [], ""

    pred = model.predict(texts=[text])[0]
    clusters = pred.get_clusters(as_strings=False)    # list of clusters of (start, end) char offsets

    if not clusters:
        return [(text, None)], "No coreference clusters found."

    spans, lines = [], []
    for k, cluster in enumerate(sorted(clusters, key=min), 1):
        cluster = sorted(cluster)
        s0, e0 = cluster[0]
        name = text[s0:e0].strip().title()
        short = name if len(name) <= 12 else name[:10] + "..."
        label = f"{k}: {short}"
        for s, e in cluster:
            spans.append((s, e, label))
        lines.append(f"Cluster {k}: " + "  |  ".join(text[s:e].strip() for s, e in cluster))

    # build the highlighted output, skipping nested/overlapping mentions
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, pos = [], 0
    for s, e, label in spans:
        if s < pos:
            continue
        if s > pos:
            out.append((text[pos:s], None))
        out.append((text[s:e], label))
        pos = e
    if pos < len(text):
        out.append((text[pos:], None))

    return out, "\n".join(lines)


css = """
body {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #e2e8f0;
}
.gradio-container {
    max-width: 900px !important; 
    margin: auto; 
    padding: 40px;
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}
h1 {
    color: #a78bfa; 
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
    text-shadow: 0 0 20px rgba(167, 139, 250, 0.4);
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
    font-size: 1.1em;
}
"""

theme = gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")
with gr.Blocks() as demo:
    gr.Markdown("# ✨ Narrative Coreference Resolver")
    gr.Markdown("<div class='subtitle'>Identify and link characters across your story using a transformer coreference model.</div>")

    with gr.Row():
        text_in = gr.Textbox(
            lines=6,
            placeholder="Paste a paragraph from a story here...",
            label="Input Story"
        )

    btn = gr.Button("🔍 Resolve Coreferences", variant="primary", size="lg")

    out = gr.HighlightedText(
        label="Resolved Mentions",
        combine_adjacent=True,
        show_legend=True,
    )
    summary = gr.Textbox(label="Clusters found", lines=6, interactive=False)

    btn.click(fn=resolve_coref, inputs=text_in, outputs=[out, summary])

    gr.Examples(
        examples=[
            ["John went to the store because he needed to buy milk. When he arrived, Mr. Smith said hello to him. He was very happy."],
            ["Alice and Bob were walking in the park. She told him that they should go home because it was getting late."],
            ["The young king looked at the ancient map. He wondered if his father had ever traveled to these distant lands himself."],
            ["As the heavy oak door creaked open, Elena gripped the edge of the table, her heart pounding with the desperate hope that her brother had finally returned unharmed, while across the room, Marcus kept his eyes fixed on the shadows in the hallway, his hand sliding slowly toward the heavy iron latch because he knew danger usually wore a quiet face."],
            ["As the sudden flash flood rushed down the narrow mountain trail, Maya scrambled desperately to secure the climbing rope to a sturdy pine, while Leo threw his weight against the heavy supply crate to keep it from sliding off the ledge, and Sam, frozen in panic for a vital second, finally lunged forward to grab their slipping communication pack before the rising muddy torrent swept it into the gorge."],
        ],
        inputs=text_in
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, css=css, theme=theme)
