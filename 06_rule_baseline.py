import json, importlib
from collections import defaultdict

loader = importlib.import_module("02_load_clusters")
metrics = importlib.import_module("05_metrics")
import os

MALE = {"he", "him", "his", "himself"}
FEMALE = {"she", "her", "hers", "herself"}
NEUTER = {"it", "its", "itself"}
PLURAL3 = {"they", "them", "their", "theirs", "themselves"}
FIRST_SG = {"i", "me", "my", "mine", "myself"}
FIRST_PL = {"we", "us", "our", "ours", "ourselves"}
SECOND = {"you", "your", "yours", "yourself", "yourselves", "thou", "thee", "thy"}
RELATIVE = {"who", "whom", "whose"}
ALL_PRON = MALE | FEMALE | NEUTER | PLURAL3 | FIRST_SG | FIRST_PL | SECOND | RELATIVE

MALE_W = {"mr", "mr.", "sir", "lord", "king", "prince", "man", "gentleman", "boy", "father",
          "brother", "son", "husband", "uncle", "master", "duke", "captain"}
FEMALE_W = {"mrs", "mrs.", "miss", "lady", "queen", "princess", "woman", "girl", "mother",
            "sister", "daughter", "wife", "aunt", "madam", "duchess"}
PLURAL_W = {"men", "women", "people", "children", "boys", "girls", "gentlemen", "ladies", "parents"}
DETS = {"the", "a", "an", "this", "that", "these", "those", "his", "her", "my", "our", "their", "its", "your"}

WINDOW = 5   # max sentence distance for 3rd-person pronouns

def pron_class(w):
    if w in MALE: return "m"
    if w in FEMALE: return "f"
    if w in NEUTER: return "n"
    if w in PLURAL3: return "p"
    if w in FIRST_SG: return "1s"
    if w in FIRST_PL: return "1p"
    if w in SECOND: return "2"
    if w in RELATIVE: return "rel"

def mention_info(words):
    low = [w.lower() for w in words]
    if len(words) == 1 and low[0] in ALL_PRON:
        return {"pron": True, "cls": pron_class(low[0])}
    gender = "m" if set(low) & MALE_W else "f" if set(low) & FEMALE_W else None
    plural = " and " in " ".join(low) or bool(set(low) & PLURAL_W)
    norm = [w for w in low if w not in DETS]
    proper = words[0][0].isupper() and words[0].lower() not in DETS
    return {"pron": False, "gender": gender, "plural": plural,
            "norm": " ".join(norm), "last": norm[-1] if norm else "", "proper": proper,
            "n_tokens": len(norm)}

def compatible(pcls, cand):
    if cand["pron"]:
        return cand["cls"] == pcls
    if pcls in ("m", "f"):
        return not cand["plural"] and cand["gender"] in (None, pcls)
    if pcls == "n":
        return not cand["plural"] and cand["gender"] is None
    if pcls == "p":
        return cand["plural"]
    return False

def predict(tokens, sents, gold_clusters):
    spans = sorted({s for c in gold_clusters.values() for s in c})
    info = [mention_info(tokens[s:e + 1]) for s, e in spans]
    parent = list(range(len(spans)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, (s, e) in enumerate(spans):
        mi = info[i]
        prev = [j for j in range(i - 1, -1, -1) if spans[j][1] < s]   # nearest first, no overlap
        link = None
        if mi["pron"]:
            c = mi["cls"]
            for j in prev:
                dist = sents[s] - sents[spans[j][1]]
                if c in ("1s", "1p"):
                    if info[j]["pron"] and info[j]["cls"] == c:
                        link = j; break
                elif c == "2":
                    if dist > WINDOW: break
                    if info[j]["pron"] and info[j]["cls"] == "2":
                        link = j; break
                elif c == "rel":
                    if dist > 0: break
                    link = j; break
                else:
                    if dist > WINDOW: break
                    if compatible(c, info[j]):
                        link = j; break
        else:
            for j in prev:
                if info[j]["pron"]:
                    continue
                same = info[j]["norm"] == mi["norm"] and mi["norm"] != ""
                partial = (mi["proper"] and info[j]["proper"] and mi["last"] == info[j]["last"]
                           and mi["n_tokens"] <= 3 and info[j]["n_tokens"] <= 3)
                if same or partial:
                    link = j; break
        if link is not None:
            parent[find(i)] = find(link)

    groups = defaultdict(set)
    for i, sp in enumerate(spans):
        groups[find(i)].add(sp)
    return list(groups.values())

def run(split_files, label):
    pairs = []
    for fname in split_files:
        tokens, sents, gold = loader.load_conll(os.path.join(loader.CONLL_DIR, fname))
        gold_sets = [set(v) for v in gold.values()]
        pairs.append((gold_sets, predict(tokens, sents, gold)))
    metrics.print_results(metrics.evaluate(pairs), label)

if __name__ == "__main__":
    splits = json.load(open("splits.json"))
    run(splits["val"], "Rule-based baseline: VALIDATION")
    run(splits["test"], "Rule-based baseline: TEST")