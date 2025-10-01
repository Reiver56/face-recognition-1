# tools/search_gallery.py
import argparse
import numpy as np
import sys
sys.path.insert(0,'/root/face-recognition-1/src')
from arcface_ov import ArcFaceOV

def cosine_matrix(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    # q: (D,), X: (N,D)  ->  (N,)
    qn = q / (np.linalg.norm(q) + 1e-12)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ qn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--gallery", required=True, help="npz prodotto da cache_embeddings.py")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--swaprb", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.54)
    args = ap.parse_args()

    data = np.load(args.gallery, allow_pickle=True)
    X = data["X"].astype(np.float32)          # (N,512)
    y = data["y"].astype(np.int64)            # (N,)
    paths = data["paths"]
    classes = data["classes"]

    model = ArcFaceOV(args.xml, swaprb=bool(args.swaprb))
    f = model.embed_path(args.query)          # (512,)

    sims = cosine_matrix(f, X)                # (N,)
    top = np.argsort(-sims)[:args.k]

    print(f"Query: {args.query}")
    for rank, idx in enumerate(top, 1):
        s = float(sims[idx])
        cls = classes[y[idx]]
        print(f"{rank:2d}. {s:.4f}  {cls}  {paths[idx]}")
    print(f"\nDecisione @τ={args.tau:.2f}: {'MATCH ✅' if sims[top[0]]>=args.tau else 'NO MATCH ❌'}")

if __name__ == "__main__":
    main()
