import os, json, random

CONLL_DIR = "litbank/coref/conll"
docs = sorted(f for f in os.listdir(CONLL_DIR) if f.endswith(".conll"))

random.seed(42)
random.shuffle(docs)

train, val, test = docs[:80], docs[80:90], docs[90:]
json.dump({"train": train, "val": val, "test": test},
          open("splits.json", "w"), indent=2)

print(len(train), len(val), len(test))