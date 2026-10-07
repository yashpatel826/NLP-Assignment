import os, re
from collections import defaultdict

CONLL_DIR = "litbank/coref/conll"

def load_conll(path):
    tokens, sent_of_token = [], []
    open_stack = defaultdict(list)   # cluster id -> stack of start indices
    clusters = defaultdict(list)     # cluster id -> list of (start, end) spans
    sent = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line.strip():
                sent += 1
                continue

            parts = line.split()
            word, coref = parts[3], parts[-1]
            idx = len(tokens)                 # global token index
            tokens.append(word)
            sent_of_token.append(sent)

            if coref == "_":
                continue
            for tag in coref.split("|"):
                m = re.fullmatch(r"(\()?(\d+)(\))?", tag)
                if not m:
                    continue
                start, cid, end = bool(m.group(1)), int(m.group(2)), bool(m.group(3))
                if start and end:
                    clusters[cid].append((idx, idx))
                elif start:
                    open_stack[cid].append(idx)
                elif end and open_stack[cid]:
                    clusters[cid].append((open_stack[cid].pop(), idx))

    return tokens, sent_of_token, dict(clusters)


if __name__ == "__main__":
    files = sorted(os.listdir(CONLL_DIR))
    tokens, sents, clusters = load_conll(os.path.join(CONLL_DIR, files[0]))

    print(files[0])
    print("tokens:", len(tokens), "| sentences:", sents[-1] + 1, "| clusters:", len(clusters))

    # show the 5 biggest clusters
    biggest = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    for cid, spans in biggest:
        mentions = [" ".join(tokens[s:e + 1]) for s, e in sorted(spans)]
        print(f"\nCluster {cid} ({len(spans)} mentions):")
        print("  ", mentions[:15])