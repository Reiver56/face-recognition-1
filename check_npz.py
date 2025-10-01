import numpy as np, os, sys, json

p = "data/index/lfw_plus_me.npz"
print(">>> Check NPZ:", p)

try:
    d = np.load(p, allow_pickle=True)
    print("Loaded OK. Keys:", list(d.keys()))
    if "embeddings" in d:
        print("embeddings.shape:", d["embeddings"].shape)
    if "labels" in d:
        labels = d["labels"]
        print("labels.len:", len(labels))
        print("labels[:3]:", labels[:3])
    if "paths" in d:
        paths = d["paths"]
        print("paths.len:", len(paths))
        print("paths[:3]:", paths[:3])
except Exception as e:
    print("ERROR loading npz:", type(e).__name__, e)
    if os.path.exists(p):
        print("file exists, size:", os.path.getsize(p))
    else:
        print("file does not exist")
    sys.exit(2)

# Check also latest_index.json
manifest = os.path.join("data", "index", "latest_index.json")
if os.path.exists(manifest):
    print("\n>>> Found manifest:", manifest)
    with open(manifest, "r", encoding="utf-8") as fh:
        print(fh.read())
else:
    print("\n>>> No manifest file found.")
