"""
Valutazione di ArcFace + OpenVINO (verification mode)
Supporta:
  1) Formato LFW classico:
     - Pos:  Name idx1 idx2
     - Neg:  Name1 idx1 Name2 idx2
  2) Formato "path-based":
     - Pos/Neg con label:    rel/path/A.jpg rel/path/B.jpg 1|0
     - Solo 2 path (senza y): rel/path/A.jpg rel/path/B.jpg  (etichetta dedotta: 1 se stessa cartella, altrimenti 0)

Output:
  - Stampa AUC, EER, TPR@FPR
  - Salva ROC e istogrammi similitudini
"""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import cv2
from arcface_ov import ArcFaceOV


def _is_int_token(t: str) -> bool:
    try:
        int(t); return True
    except:
        return False


from pathlib import Path

def load_pairs(pairs_file, root):
    """
    Supporta formati:
      (A) LFW classico con indici:
          Name idx1 idx2            -> positivo
          Name1 idx1 Name2 idx2     -> negativo
      (B) Variante con filenames:
          Name file1.jpg file2.jpg                  (pos)
          Name1 file1.jpg Name2 file2.jpg           (neg)
      (C) Path-based:
          rel/path1.jpg rel/path2.jpg [0|1]
    Salta l'header tipo "2605 3". Verifica esistenza dei file.
    """
    root = Path(root)
    pairs = []
    missing = 0

    with open(pairs_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()

            # header tipo "2605 3"
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue

            p1 = p2 = None
            label = None

            # --- (C) path-based ---
            if len(parts) in (2, 3) and (parts[0].endswith(".jpg") or "/" in parts[0] or "\\" in parts[0]):
                p1 = root / parts[0]
                p2 = root / parts[1]
                if len(parts) == 3 and parts[2] in ("0", "1"):
                    label = int(parts[2])
                else:
                    # deduzione semplice: stessa cartella -> positivo
                    label = 1 if Path(parts[0]).parent == Path(parts[1]).parent else 0

            # --- (B) filenames (positivo) ---
            elif len(parts) == 3 and parts[1].lower().endswith(".jpg") and parts[2].lower().endswith(".jpg"):
                name, f1, f2 = parts
                p1 = root / name / f1
                p2 = root / name / f2
                label = 1

            # --- (B) filenames (negativo) ---
            elif len(parts) == 4 and parts[1].lower().endswith(".jpg") and parts[3].lower().endswith(".jpg"):
                n1, f1, n2, f2 = parts
                p1 = root / n1 / f1
                p2 = root / n2 / f2
                label = 0

            # --- (A) indici LFW (positivo) ---
            elif len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                name, i1, i2 = parts
                p1 = root / name / f"{name}_{int(i1):04d}.jpg"
                p2 = root / name / f"{name}_{int(i2):04d}.jpg"
                label = 1

            # --- (A) indici LFW (negativo) ---
            elif len(parts) == 4 and parts[1].isdigit() and parts[3].isdigit():
                n1, i1, n2, i2 = parts
                p1 = root / n1 / f"{n1}_{int(i1):04d}.jpg"
                p2 = root / n2 / f"{n2}_{int(i2):04d}.jpg"
                label = 0

            # se non riconosciuto, passa oltre
            if p1 is None or p2 is None or label is None:
                # print("[skip] formato non riconosciuto:", parts)
                continue

            # verifica esistenza file
            if not p1.exists() or not p2.exists():
                # print("[missing]", p1, "|", p2)
                missing += 1
                continue

            pairs.append((str(p1), str(p2), label))

    if missing:
        print(f"[warn] coppie saltate per file mancanti: {missing}")
    print(f"[info] coppie caricate valide: {len(pairs)}")
    # quick breakdown
    pos = sum(1 for _,_,y in pairs if y==1)
    neg = sum(1 for _,_,y in pairs if y==0)
    print(f"[info] breakdown -> pos={pos}, neg={neg}")
    return pairs



def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/arcface_openvino.json")
    ap.add_argument("--pairs", required=True, help="file pairs.txt")
    ap.add_argument("--root", required=True, help="root immagini (es. data/lfw o data/enroll)")
    ap.add_argument("--preproc", default="raw255", choices=["raw255","arcface"], help="preprocessing embedder")
    ap.add_argument("--save-dir", default="figs", help="cartella per le figure")
    #ap.add_argument("--save-roc", default="figs", help="cartella per i dati ROC/similarità")
    args = ap.parse_args()

    # modello
    cfg = json.loads(Path(args.config).read_text())
    arc = ArcFaceOV(cfg["xml"], swaprb=cfg.get("swaprb", False), preproc=args.preproc)

    # pairs
    pairs = load_pairs(args.pairs, args.root)
    if not pairs:
        raise RuntimeError(f"Nessuna coppia valida trovata in {args.pairs}")
    print(f"[info] coppie caricate: {len(pairs)}")

    # calcolo similitudini
    sims, labels = [], []
    missing = 0
    for i, (p1, p2, y) in enumerate(pairs):
        im1 = cv2.imread(p1)
        im2 = cv2.imread(p2)
        if im1 is None or im2 is None:
            missing += 1
            continue
        f1 = arc.embed_bgr(im1)
        f2 = arc.embed_bgr(im2)
        sims.append(cosine_sim(f1, f2))
        labels.append(y)
        if (i+1) % 1000 == 0:
            print(f"[info] processed {i+1}/{len(pairs)}")

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    #np.savez("figs/sims_data.npz", sims=sims, labels=labels)

    if sims.size == 0:
        raise RuntimeError("Nessuna similitudine calcolata (quasi certamente file mancanti). Controlla i path in pairs.txt.")

    # Deve contenere sia positivi che negativi
    uniq = set(labels.tolist())
    if not ({0,1} <= uniq or uniq == {0,1}):
        raise RuntimeError(f"Le etichette contengono una sola classe: {uniq}. Impossibile calcolare ROC/EER. Verifica pairs.txt.")


    # metriche
    fpr, tpr, thr = metrics.roc_curve(labels, sims)
    auc = metrics.auc(fpr, tpr)
    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    def tpr_at_fpr(target):
        idx = np.where(fpr <= target)[0]
        return float(tpr[idx[-1]]) if len(idx) else 0.0

    tpr1e3 = tpr_at_fpr(1e-3)
    tpr1e4 = tpr_at_fpr(1e-4)

    print(f"AUC={auc:.4f} | EER={eer:.4f} | TPR@1e-3={tpr1e3:.3f} | TPR@1e-4={tpr1e4:.3f}")
    if missing:
        print(f"[warn] immagini mancanti: {missing}")

    # Salva i risultati grezzi (similarità + etichette) per analisi successive
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "sims_data.npz", sims=sims, labels=labels)

    # figure
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=150)

    plt.figure()
    plt.hist(sims[labels == 1], bins=50, alpha=0.6, label="Positivi")
    plt.hist(sims[labels == 0], bins=50, alpha=0.6, label="Negativi")
    plt.xlabel("Cosine similarity"); plt.ylabel("Frequenza")
    plt.title("Distribuzione delle similarità")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "hist_similarity.png", dpi=150)

    print(f"[done] grafici salvati in: {outdir}/")


if __name__ == "__main__":
    main()
