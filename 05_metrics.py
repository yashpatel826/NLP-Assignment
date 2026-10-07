import numpy as np
from scipy.optimize import linear_sum_assignment

def muc(gold, pred):
    def score(A, B):
        m2c = {m: i for i, c in enumerate(B) for m in c}
        num = den = 0
        for c in A:
            parts = {m2c.get(m, ("s", m)) for m in c}
            num += len(c) - len(parts)
            den += len(c) - 1
        return num, den
    rn, rd = score(gold, pred)
    pn, pd = score(pred, gold)
    return rn, rd, pn, pd

def b3(gold, pred):
    g_of = {m: c for c in gold for m in c}
    p_of = {m: c for c in pred for m in c}
    rn = sum(len(c & p_of.get(m, {m})) / len(c) for m, c in g_of.items())
    pn = sum(len(c & g_of.get(m, {m})) / len(c) for m, c in p_of.items())
    return rn, len(g_of), pn, len(p_of)

def ceafe(gold, pred):
    if not gold or not pred:
        return 0, len(gold), 0, len(pred)
    sim = np.array([[2 * len(g & p) / (len(g) + len(p)) for p in pred] for g in gold])
    r, c = linear_sum_assignment(-sim)
    tot = sim[r, c].sum()
    return tot, len(gold), tot, len(pred)

METRICS = {"MUC": muc, "B3": b3, "CEAFe": ceafe}

def evaluate(pairs):
    """pairs: list of (gold_clusters, pred_clusters); each cluster is a set of (start, end) spans."""
    tot = {k: [0.0, 0.0, 0.0, 0.0] for k in METRICS}
    for gold, pred in pairs:
        gold = [frozenset(c) for c in gold if len(c) > 1]
        pred = [frozenset(c) for c in pred if len(c) > 1]
        for name, fn in METRICS.items():
            for i, v in enumerate(fn(gold, pred)):
                tot[name][i] += v
    out = {}
    for name, (rn, rd, pn, pd) in tot.items():
        r = rn / rd if rd else 0.0
        p = pn / pd if pd else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = (p, r, f)
    out["CoNLL F1"] = sum(out[k][2] for k in METRICS) / 3
    return out

def print_results(res, title=""):
    print(f"\n{title}")
    print(f"{'Metric':<10}{'P':>8}{'R':>8}{'F1':>8}")
    for k in METRICS:
        p, r, f = res[k]
        print(f"{k:<10}{p*100:8.1f}{r*100:8.1f}{f*100:8.1f}")
    print(f"{'CoNLL F1':<10}{'':>16}{res['CoNLL F1']*100:8.1f}")

if __name__ == "__main__":
    # sanity check: a perfect prediction must score 100
    g = [{(0, 0), (5, 5), (9, 9)}, {(2, 3), (7, 7)}]
    print_results(evaluate([(g, g)]), "Self-test (should be all 100.0)")