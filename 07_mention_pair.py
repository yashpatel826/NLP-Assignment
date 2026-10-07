import os, json, importlib
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

loader = importlib.import_module("02_load_clusters")
metrics = importlib.import_module("05_metrics")
rb = importlib.import_module("06_rule_baseline")

MAX_CAND = 30   # candidate antecedents per mention

# ---------- features ----------
def pair_features(i, j, spans, info, sents):
    (si, ei), (sj, ej) = spans[i], spans[j]
    mi, mj = info[i], info[j]
    sd = sents[si] - sents[ej]
    f = {
        "sent_dist": min(sd, 20) / 20,
        "log_tok_dist": np.log1p(max(0, si - ej)) / 7,
        "ment_dist": min(i - j, MAX_CAND) / MAX_CAND,
        "same_sent": int(sd == 0),
        "i_pron": int(mi["pron"]), "j_pron": int(mj["pron"]),
        "i_cls": mi["cls"] if mi["pron"] else "np",
        "j_cls": mj["cls"] if mj["pron"] else "np",
        "i_len": np.log1p(ei - si + 1), "j_len": np.log1p(ej - sj + 1),
    }
    if mi["pron"] and mi["cls"] in ("m", "f", "n", "p"):
        f["compat"] = "yes" if rb.compatible(mi["cls"], mj) else "no"
    else:
        f["compat"] = "na"
    f["both_pron_same_cls"] = int(mi["pron"] and mj["pron"] and mi["cls"] == mj["cls"])
    if not mi["pron"]:
        f["i_proper"] = int(mi["proper"]); f["i_gender"] = mi["gender"] or "none"
        f["i_plural"] = int(mi["plural"])
    if not mj["pron"]:
        f["j_proper"] = int(mj["proper"]); f["j_gender"] = mj["gender"] or "none"
        f["j_plural"] = int(mj["plural"])
    if not mi["pron"] and not mj["pron"]:
        f["exact_match"] = int(mi["norm"] == mj["norm"] and mi["norm"] != "")
        f["head_match"] = int(mi["last"] == mj["last"] and mi["last"] != "")
        wi, wj = set(mi["norm"].split()), set(mj["norm"].split())
        f["word_overlap"] = len(wi & wj) / max(1, len(wi | wj))
        f["both_proper"] = int(mi["proper"] and mj["proper"])
        f["gender_agree"] = ("na" if None in (mi["gender"], mj["gender"])
                             else "yes" if mi["gender"] == mj["gender"] else "no")
    return f

# ---------- data ----------
def prepare(files):
    data = []
    for fname in files:
        tokens, sents, gold = loader.load_conll(os.path.join(loader.CONLL_DIR, fname))
        span_cid = {}
        for cid, sp_list in gold.items():
            for sp in sp_list:
                span_cid.setdefault(sp, cid)
        spans = sorted(span_cid)
        info = [rb.mention_info(tokens[s:e + 1]) for s, e in spans]
        feats, ij, y = [], [], []
        for i in range(len(spans)):
            c = 0
            for j in range(i - 1, -1, -1):
                if spans[j][1] >= spans[i][0]:      # skip overlapping/nested
                    continue
                feats.append(pair_features(i, j, spans, info, sents))
                ij.append((i, j))
                y.append(int(span_cid[spans[i]] == span_cid[spans[j]]))
                c += 1
                if c >= MAX_CAND:
                    break
        data.append(dict(spans=spans, gold=gold, feats=feats, ij=ij, y=y))
    return data

def decode(d, probs, thr):
    n = len(d["spans"])
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    best = {}
    for (i, j), p in zip(d["ij"], probs):
        if i not in best or p > best[i][0]:
            best[i] = (p, j)
    for i, (p, j) in best.items():
        if p >= thr:
            parent[find(i)] = find(j)
    groups = defaultdict(set)
    for i, sp in enumerate(d["spans"]):
        groups[find(i)].add(sp)
    return list(groups.values())

def score(data, all_probs, thr):
    pairs = [([set(v) for v in d["gold"].values()], decode(d, p, thr))
             for d, p in zip(data, all_probs)]
    return metrics.evaluate(pairs)

# ---------- run ----------
if __name__ == "__main__":
    splits = json.load(open("splits.json"))
    train, val, test = (prepare(splits[k]) for k in ("train", "val", "test"))

    vec = DictVectorizer(sparse=False, dtype=np.float32)
    Xtr = vec.fit_transform([f for d in train for f in d["feats"]])
    ytr = np.array([y for d in train for y in d["y"]])
    print(f"Training pairs: {len(ytr)} | positive share: {ytr.mean():.1%} | features: {Xtr.shape[1]}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42),
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        val_p = [model.predict_proba(vec.transform(d["feats"]))[:, 1] for d in val]
        test_p = [model.predict_proba(vec.transform(d["feats"]))[:, 1] for d in test]

        # choose the link threshold on VALIDATION only
        best_thr, best_f = 0.5, -1
        for thr in np.arange(0.2, 0.85, 0.05):
            f = score(val, val_p, thr)["CoNLL F1"]
            if f > best_f:
                best_thr, best_f = thr, f
        print(f"\n[{name}] best threshold on validation: {best_thr:.2f}")
        metrics.print_results(score(val, val_p, best_thr), f"{name}: VALIDATION")
        metrics.print_results(score(test, test_p, best_thr), f"{name}: TEST")

        if name == "Logistic Regression":
            names = vec.get_feature_names_out()
            order = np.argsort(model.coef_[0])
            print("\nStrongest features FOR coreference:")
            for k in order[::-1][:10]:
                print(f"  {names[k]:<28}{model.coef_[0][k]:+.2f}")
            print("Strongest features AGAINST coreference:")
            for k in order[:10]:
                print(f"  {names[k]:<28}{model.coef_[0][k]:+.2f}")