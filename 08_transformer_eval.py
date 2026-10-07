"""
09_transformer_eval.py  -- evaluate pretrained transformer coreference models on LitBank

Run on Google Colab (GPU):
  1) Runtime > Change runtime type > T4 GPU
  2) Upload to /content:  02_load_clusters.py, 05_metrics.py, 09_transformer_eval.py, splits.json
  3) Cell:  !pip install -q fastcoref
            !git clone -q https://github.com/dbamman/litbank.git
  4) Cell:  !python 09_transformer_eval.py

If you get import/attribute errors from `transformers`:
            !pip install -q "transformers==4.38.2"   then Runtime > Restart session, run again.

Outputs: printed tables + transformer_results.json
"""
import os, json, importlib
from collections import defaultdict
import torch
import transformers
transformers.PreTrainedModel.all_tied_weights_keys = {}
from fastcoref import FCoref, LingMessCoref

loader = importlib.import_module("02_load_clusters")
metrics = importlib.import_module("05_metrics")

SPLIT = "test"      # "test" or "val"

NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "'s", "n't", "'ll",
                   "'re", "'ve", "'d", "'m", "’s", "n’t"}
OPENERS = {"(", "[", "{"}
QUOTES = {'"', "``", "''", "“", "”"}

MODES = {
    "gold_exact":   "Gold mentions, exact match (comparable to the other models)",
    "gold_relaxed": "Gold mentions, head-word match (lenient on boundaries)",
    "end2end":      "End-to-end, model finds its own mentions (strictest)",
}

SAMPLE_1 = ("As the heavy oak door creaked open, Elena gripped the edge of the table, "
            "her heart pounding with the desperate hope that her brother had finally "
            "returned unharmed, while across the room, Marcus kept his eyes fixed on "
            "the shadows in the hallway, his hand sliding slowly toward the heavy iron "
            "latch because he knew danger usually wore a quiet face.")
SAMPLE_2 = ("As the sudden flash flood rushed down the narrow mountain trail, Maya scrambled "
            "desperately to secure the climbing rope to a sturdy pine, while Leo threw his weight "
            "against the heavy supply crate to keep it from sliding off the ledge, and Sam, frozen "
            "in panic for a vital second, finally lunged forward to grab their slipping "
            "communication pack before the rising muddy torrent swept it into the gorge.")


# ---------------- text <-> token offsets ----------------
def build_text(tokens):
    """Join tokens into natural text and remember each token's character offsets."""
    text, offsets = "", []
    in_quote, glue_next = False, False
    for tok in tokens:
        space = bool(text) and not glue_next
        glue_next = False
        if tok in NO_SPACE_BEFORE:
            space = False
        elif tok in QUOTES:
            if in_quote:
                space = False            # closing quote
            else:
                glue_next = True         # opening quote
            in_quote = not in_quote
        elif tok in OPENERS:
            glue_next = True
        if space:
            text += " "
        offsets.append((len(text), len(text) + len(tok)))
        text += tok
    return text, offsets


def char_to_token_span(s, e_excl, offsets):
    idx = [i for i, (a, b) in enumerate(offsets) if b > s and a < e_excl]
    return (idx[0], idx[-1]) if idx else None


def detect_end_offset(model):
    """Check whether fastcoref's character spans have an exclusive (0) or inclusive (1) end."""
    p = model.predict(texts=[SAMPLE_1])[0]
    flat_s = [x for cl in p.get_clusters(as_strings=True) for x in cl]
    flat_i = [x for cl in p.get_clusters(as_strings=False) for x in cl]
    if not flat_s:
        print("WARNING: could not verify offsets (no clusters on sample); assuming exclusive end.")
        return 0
    if all(SAMPLE_1[s:e] == t for (s, e), t in zip(flat_i, flat_s)):
        return 0
    if all(SAMPLE_1[s:e + 1] == t for (s, e), t in zip(flat_i, flat_s)):
        return 1
    raise RuntimeError("Character offsets from fastcoref do not match its strings; check the library version.")


# ---------------- data ----------------
def load_split(name):
    files = json.load(open("splits.json"))[name]
    docs = []
    for f in files:
        tokens, sents, gold = loader.load_conll(os.path.join(loader.CONLL_DIR, f))
        text, offsets = build_text(tokens)
        docs.append(dict(
            name=f, text=text, offsets=offsets,
            gold_sets=[set(v) for v in gold.values()],
            gold_spans={sp for v in gold.values() for sp in v},
        ))
    return docs


def predict_clusters(model, docs, end_offset):
    preds = model.predict(texts=[d["text"] for d in docs])
    all_clusters = []
    for d, p in zip(docs, preds):
        clusters = []
        for cl in p.get_clusters(as_strings=False):
            spans = {char_to_token_span(s, e + end_offset, d["offsets"]) for s, e in cl}
            spans.discard(None)
            if spans:
                clusters.append(spans)
        all_clusters.append(clusters)
    return all_clusters


# ---------------- evaluation ----------------
def to_gold_mentions(clusters, gold_spans, relaxed):
    """Keep only predicted mentions that match a gold mention.
    exact   = same start and end token
    relaxed = same last token (head word), closest start"""
    by_end = defaultdict(list)
    for sp in gold_spans:
        by_end[sp[1]].append(sp)
    used, out = set(), []
    for cl in clusters:
        new = set()
        for s, e in cl:
            if (s, e) in gold_spans:
                m = (s, e)
            elif relaxed and e in by_end:
                m = min(by_end[e], key=lambda g: abs(g[0] - s))
            else:
                continue
            if m not in used:
                used.add(m)
                new.add(m)
        if new:
            out.append(new)
    return out


def evaluate_model(name, docs, all_clusters):
    summary = {}
    for mode, desc in MODES.items():
        pairs = []
        for d, clusters in zip(docs, all_clusters):
            pred = clusters if mode == "end2end" else to_gold_mentions(
                clusters, d["gold_spans"], relaxed=(mode == "gold_relaxed"))
            pairs.append((d["gold_sets"], pred))
        res = metrics.evaluate(pairs)
        metrics.print_results(res, f"{name} | {desc}")
        summary[mode] = {k: ([round(x * 100, 1) for x in v] if isinstance(v, tuple) else round(v * 100, 1))
                         for k, v in res.items()}
    return summary


def show_clusters(model, text, title):
    print(f"\n--- {title} ---")
    p = model.predict(texts=[text])[0]
    for k, cl in enumerate(p.get_clusters(), 1):
        print(f"Cluster {k}: {cl}")


# ---------------- main ----------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    docs = load_split(SPLIT)
    print(f"{len(docs)} stories ({SPLIT}), {sum(len(d['offsets']) for d in docs)} tokens")

    results = {}
    for name, cls in [("F-Coref", FCoref), ("LingMess", LingMessCoref)]:
        print(f"\n===== {name} =====")
        model = cls(device=device)
        end_offset = detect_end_offset(model)
        print("Offset check passed (end offset =", end_offset, ")")
        clusters = predict_clusters(model, docs, end_offset)
        results[name] = evaluate_model(name, docs, clusters)
        show_clusters(model, SAMPLE_1, f"{name}: Elena / Marcus")
        show_clusters(model, SAMPLE_2, f"{name}: Maya / Leo / Sam")
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    json.dump(results, open("transformer_results.json", "w"), indent=2)

    print("\n===== SUMMARY (CoNLL F1) =====")
    if SPLIT == "test":
        print("Earlier models (test, gold mentions): Rule-based 57.3 | Logistic Regression 60.4 | Gradient Boosting 63.4")
    for name, res in results.items():
        print(f"{name:10s} " + " | ".join(f"{m}: {res[m]['CoNLL F1']}" for m in MODES))
    print("\nSaved transformer_results.json")


if __name__ == "__main__":
    main()