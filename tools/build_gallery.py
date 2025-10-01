#!/usr/bin/env python3
import argparse, sys, json
from pathlib import Path
import numpy as np
import cv2

# permetti "from src.arcface_ov import ArcFaceOV"
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from arcface_ov import ArcFaceOV  # già creato prima

def iter_images_labeled(root: Path):
    """
    root/
      person_a/*.jpg
      person_b/*.png
    """
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name
        for imgp in sorted(cls_dir.rglob("*")):
            if imgp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                yield imgp, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/arcface_openvino.json")
    ap.add_argument("--root", required=True, help="Root con sottocartelle per identità")
    ap.add_argument("--out", required=True, help="Path output .npz (indice)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    arc = ArcFaceOV(cfg["xml"], swaprb=cfg.get("swaprb", False), preproc=cfg.get("preproc", "raw255"))

    embs = []
    labels = []
    paths  = []

    total = 0
    for imgp, lab in iter_images_labeled(Path(args.root)):
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        f = arc.embed_bgr(img)
        embs.append(f)
        labels.append(lab)
        paths.append(str(imgp))
        total += 1
        if total % 200 == 0:
            print(f"[progress] {total} immagini...")

    if not embs:
        raise RuntimeError("Nessuna immagine valida trovata.")

    E = np.vstack(embs).astype(np.float32)
    uniq = sorted(set(labels))
    print(f"[done] embeddings: {E.shape}, identità: {len(uniq)}")

    np.savez_compressed(args.out, embeddings=E, labels=np.array(labels, dtype=object), paths=np.array(paths, dtype=object))
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
