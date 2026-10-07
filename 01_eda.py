import os

path = "litbank/coref/conll"   # adjust to where your folder is
files = sorted(os.listdir(path))
print(len(files), "files")
print(files[:5])

with open(os.path.join(path, files[0]), encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i > 25:
            break