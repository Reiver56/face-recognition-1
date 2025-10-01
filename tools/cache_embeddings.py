import argparse
from pathlib import Path
import numpy as np
import cv2
import sys
sys.path.insert(0,'/root/face-recognition-1/src')
from arcface_ov import ArcFaceOV

def walk_aligned(root: Path):
    classes = [d for d in sorted(root.iterdir()) if d.is_dir()]
    class_names = [d.name for d in classes]
    paths, labels = [], []
    for lid, d in enumerate(classes):
        for p in sorted(d.glob("*.*")):
            paths.append(str(p))
            labels.append(lid)
    return paths, np.array(labels, dtype=np.int64), class_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--root", required=True, help="cartella con sottocartelle per identità (es. data/aligned/test)")
    ap.add_argument("--out", required=True, help="file npz di output (es. models/gallery.npz)")
    ap.add_argument("--swaprb", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    paths, y, class_names = walk_aligned(root)
    if not paths:
        raise SystemExit(f"Nessuna immagine trovata in {root}")

    model = ArcFaceOV(args.xml, swaprb=bool(args.swaprb))

    X = []
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] salto (illeggibile): {p}")
            continue
        X.append(model.embed_bgr(img))
        if i % 50 == 0:
            print(f"[{i}/{len(paths)}]")

    X = np.vstack(X).astype(np.float32)
    np.savez_compressed(args.out, X=X, y=y, paths=np.array(paths), classes=np.array(class_names))
    print(f"[OK] Salvato {args.out}  (X={X.shape}, n_class={len(class_names)})")

if __name__ == "__main__":
    main()
