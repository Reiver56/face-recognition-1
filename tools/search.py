#!/usr/bin/env python3
import argparse, sys, json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from arcface_ov import ArcFaceOV

def cosine_matrix(q: np.ndarray, E: np.ndarray) -> np.ndarray:
    q = q.reshape(1, -1).astype(np.float32)
    denom = (np.linalg.norm(E, axis=1) * np.linalg.norm(q) + 1e-12)
    return (E @ q.T).ravel() / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tau", type=float, default=None, help="Soglia. Default: usa 'thresholds.default' da config")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tau = args.tau if args.tau is not None else float(cfg["thresholds"]["default"])

    data = np.load(args.index, allow_pickle=True)
    E = data["embeddings"].astype(np.float32)
    labels = data["labels"]
    paths = data["paths"]

    model = ArcFaceOV(cfg["xml"], swaprb=cfg.get("swaprb", False), preproc=cfg.get("preproc", "raw255"))

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)
    f = model.embed_bgr(img)

    s = cosine_matrix(f, E)
    order = np.argsort(-s)[:args.topk]
    print(f"[query] {args.img}")
    print(f"[tau] {tau:.3f}")
    print("Top-k risultati:")
    for rank, i in enumerate(order, 1):
        print(f"{rank:2d}. s={s[i]:.4f}  label={labels[i]}  path={paths[i]}")
    s1, lab1 = float(s[order[0]]), str(labels[order[0]])
    pred = lab1 if s1 >= tau else "UNKNOWN"
    print(f"\nPredizione: {pred}  (score={s1:.4f})")

if __name__ == "__main__":
    main()