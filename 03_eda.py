import os, importlib
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

loader = importlib.import_module("02_load_clusters")
load_conll, CONLL_DIR = loader.load_conll, loader.CONLL_DIR

PRONOUNS = {"he","him","his","himself","she","her","hers","herself",
            "it","its","itself","they","them","their","theirs","themselves",
            "i","me","my","mine","myself","we","us","our","ours","ourselves",
            "you","your","yours","yourself","who","whom","whose"}
DETERMINERS = {"the","a","an","his","her","my","our","their","its","your","this","that"}

os.makedirs("eda_outputs", exist_ok=True)

def mention_type(words):
    first = words[0]
    if len(words) == 1:
        if first.lower() in PRONOUNS:
            return "pronoun"
        return "proper" if first[0].isupper() else "nominal"
    if first.lower() in DETERMINERS:
        return "nominal"
    return "proper" if first[0].isupper() else "nominal"

doc_rows, mention_rows, dist_rows = [], [], []

for fname in sorted(os.listdir(CONLL_DIR)):
    if not fname.endswith(".conll"):
        continue
    doc = fname.replace("_brat.conll", "")
    tokens, sents, clusters = load_conll(os.path.join(CONLL_DIR, fname))

    doc_rows.append(dict(
        doc=doc, tokens=len(tokens), sentences=sents[-1] + 1,
        clusters=len(clusters),
        clusters_ge2=sum(1 for s in clusters.values() if len(s) >= 2),
        mentions=sum(len(s) for s in clusters.values()),
    ))

    for cid, spans in clusters.items():
        spans = sorted(spans)
        types = [mention_type(tokens[s:e + 1]) for s, e in spans]
        for i, ((s, e), t) in enumerate(zip(spans, types)):
            mention_rows.append(dict(
                doc=doc, cid=cid, cluster_size=len(spans), type=t, length=e - s + 1,
                pronoun=" ".join(tokens[s:e + 1]).lower() if t == "pronoun" else None,
            ))
            if t == "pronoun":
                prev = [j for j in range(i) if types[j] != "pronoun"]
                if prev:  # distance to nearest earlier non-pronoun mention
                    ps, pe = spans[prev[-1]]
                    dist_rows.append(dict(tok_dist=max(0, s - pe),
                                          sent_dist=sents[s] - sents[pe]))

docs = pd.DataFrame(doc_rows)
ments = pd.DataFrame(mention_rows)
dists = pd.DataFrame(dist_rows)

# ---------- summary numbers ----------
cluster_sizes = ments.groupby(["doc", "cid"]).size()
print("Documents:", len(docs))
print("Tokens:", docs.tokens.sum(), "| Sentences:", docs.sentences.sum())
print("Mentions:", len(ments))
print("Clusters (all):", docs.clusters.sum(), "| with >=2 mentions:", docs.clusters_ge2.sum())
print("Singleton share: {:.1%}".format((cluster_sizes == 1).mean()))
print("Avg clusters/doc: {:.1f} | Avg mentions/doc: {:.1f}".format(docs.clusters.mean(), docs.mentions.mean()))
print("\nMention types:\n", ments.type.value_counts())
print("\nMention length (tokens):\n", ments.length.describe())
print("\nPronoun->antecedent sentence distance:\n", dists.sent_dist.describe())
print("Share within same/previous sentence: {:.1%}".format((dists.sent_dist <= 1).mean()))

docs.to_csv("eda_outputs/per_document_stats.csv", index=False)

# ---------- plots ----------
def save(name):
    plt.tight_layout()
    plt.savefig(f"eda_outputs/{name}.png", dpi=200)
    plt.close()

ments.type.value_counts().plot(kind="bar", color="#4C72B0")
plt.title("Mention type distribution"); plt.ylabel("Count"); plt.xticks(rotation=0)
save("mention_types")

sizes = cluster_sizes[cluster_sizes >= 2].clip(upper=30)
plt.hist(sizes, bins=range(2, 32), color="#55A868")
plt.title("Cluster size (clusters with >=2 mentions, capped at 30)")
plt.xlabel("Mentions per cluster"); plt.ylabel("Clusters")
save("cluster_sizes")

plt.hist(dists.sent_dist.clip(upper=15), bins=range(0, 17), color="#C44E52")
plt.title("Sentence distance from pronoun to nearest earlier name/noun mention")
plt.xlabel("Sentences (capped at 15)"); plt.ylabel("Pronouns")
save("pronoun_distance")

ments.pronoun.value_counts().head(12).plot(kind="bar", color="#8172B2")
plt.title("Most frequent pronouns"); plt.ylabel("Count")
save("top_pronouns")

plt.hist(ments.length.clip(upper=12), bins=range(1, 14), color="#CCB974")
plt.title("Mention length (tokens, capped at 12)")
plt.xlabel("Tokens"); plt.ylabel("Mentions")
save("mention_length")

print("\nSaved plots and CSV in eda_outputs/")