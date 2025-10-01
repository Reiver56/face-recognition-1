# src/eval_verify_openvino.py
import argparse
from pathlib import Path
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from openvino.runtime import Core

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "aligned"

# -----------------------
# util I/O e layout
# -----------------------
def dims_from_ps(ps):
    out = []
    for d in ps:
        out.append(int(d.get_length()) if d.is_static else -1)
    return out

def detect_layout(inp_port) -> str:
    dims = dims_from_ps(inp_port.partial_shape)
    layout = "NCHW"
    if len(dims) == 4:
        c, last = dims[1], dims[-1]
        if (c not in (1, 3)) and (last in (1, 3)):
            layout = "NHWC"
    return layout

def pick_embedding_output(model):
    """
    Sceglie un output plausibile per l'embedding:
    - preferisce 2D con ultima dim in {512, 256}
    - altrimenti 4D che contenga 512/256 tra le dimensioni (poi faremo GAP)
    - fallback: primo output
    """
    cand2d, cand4d = [], []
    for o in model.outputs:
        sh = dims_from_ps(o.partial_shape)
        if len(sh) == 2 and sh[-1] in (512, 256):
            cand2d.append((o, sh))
        elif len(sh) == 4 and (512 in sh or 256 in sh):
            cand4d.append((o, sh))
    if cand2d:
        return cand2d[0][0], "2d"
    if cand4d:
        return cand4d[0][0], "4d"
    return model.outputs[0], "unknown"

def load_ov_session(xml_path: Path):
    core = Core()
    model = core.read_model(str(xml_path))

    print("== OpenVINO model I/O ==")
    for i, inp in enumerate(model.inputs):
        print(f"Input[{i}]: name={inp.get_any_name()} shape={inp.partial_shape}")
    for i, out in enumerate(model.outputs):
        print(f"Output[{i}]: name={out.get_any_name()} shape={out.partial_shape}")

    out_port, out_kind = pick_embedding_output(model)
    print(f"[pick] Using output '{out_port.get_any_name()}' kind={out_kind}")

    compiled = core.compile_model(model, "CPU")
    inp_port = compiled.input(0)  # port del compiled
    layout = detect_layout(model.inputs[0])
    return compiled, layout, inp_port, compiled.output(out_port.get_index()), out_kind

# -----------------------
# preprocessing + embed
# -----------------------
def make_blob(img_bgr, preproc: str, swaprb: bool):
    img = cv2.resize(img_bgr, (112, 112))
    preproc = preproc.lower()

    if preproc == "raw255":
        # BGR 0..255 (OMZ IR expects this) — mean=0, scale=1
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(112, 112),
            mean=(0, 0, 0), swapRB=swaprb, crop=False
        ).astype(np.float32)

    elif preproc == "arcface":
        # (x - 127.5)/127.5 con eventuale swapRB
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0/127.5, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=swaprb, crop=False
        ).astype(np.float32)

    elif preproc == "imagenet":
        # x/255, poi (x-mean)/std canale per canale
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0/255.0, size=(112, 112),
            mean=(0, 0, 0), swapRB=swaprb, crop=False
        ).astype(np.float32)
        # mean/std in ordine del blob (NCHW con BGR->RGB già fatto da swapRB se True)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)
        blob = (blob - mean) / std
    else:
        raise ValueError(f"preproc sconosciuto: {preproc}")

    return blob

def embed(compiled, layout, inp_port, out_port, out_kind, img_bgr, preproc: str, swaprb: bool):
    x = make_blob(img_bgr, preproc=preproc, swaprb=swaprb)                # (1,3,112,112)
    if layout == "NHWC":
        x = np.transpose(x, (0, 2, 3, 1))                                 # (1,112,112,3)

    req = compiled.create_infer_request()
    res = req.infer({inp_port: x})
    out = res[out_port]

    # Normalizza a (D,)
    if out.ndim == 4:         # (N,C,H,W) -> GAP
        out = out.mean(axis=(2, 3))
    elif out.ndim == 3:       # (N,C,H) -> media su H
        out = out.mean(axis=2)

    feat = out.reshape(-1).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-12)
    return feat

def load_split(compiled, layout, inp_port, out_port, out_kind, split="test", preproc="raw255", swaprb=True):
    X, y, paths = [], [], []
    root = ALIGNED / split
    classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    for lid, cname in enumerate(classes):
        for p in sorted((root/cname).glob("*.*")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            X.append(embed(compiled, layout, inp_port, out_port, out_kind, img, preproc, swaprb))
            y.append(lid)
            paths.append(str(p))
    if len(X) == 0:
        return np.empty((0, 512), dtype=np.float32), np.array([]), [], classes
    return np.vstack(X), np.array(y), paths, classes

# -----------------------
# metriche
# -----------------------
def cosine(a, b):
    return (a*b).sum(axis=1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1) + 1e-12)

def build_pairs_safe(y, max_pos=2000, max_neg=2000, seed=42):
    rs = np.random.RandomState(seed)
    y = np.asarray(y)
    n = len(y)
    idx_by_c = {c: np.where(y == c)[0] for c in np.unique(y)}
    pos_all = []
    for _, idx in idx_by_c.items():
        if len(idx) >= 2:
            pos_all += list(combinations(idx.tolist(), 2))
    neg_all = []
    for i, j in combinations(range(n), 2):
        if y[i] != y[j]:
            neg_all.append((i, j))
    rs.shuffle(pos_all); rs.shuffle(neg_all)
    pos = np.array(pos_all[:max_pos], dtype=np.int64) if pos_all else np.empty((0,2), dtype=np.int64)
    neg = np.array(neg_all[:max_neg], dtype=np.int64) if neg_all else np.empty((0,2), dtype=np.int64)
    return pos, neg

def tpr_tau_at_fpr_interp(fpr, tpr, thr, fpr_target):
    """
    Interpola TPR e soglia τ al FPR target (evita τ=inf quando i punti ROC sono pochi).
    """
    mask = np.isfinite(thr)
    f = fpr[mask]
    t = tpr[mask]
    th = thr[mask]
    if len(f) == 0:
        return 0.0, float("nan")
    # f è crescente (roc_curve)
    tpr_t = float(np.interp(fpr_target, f, t))
    tau_t = float(np.interp(fpr_target, f, th))
    return tpr_t, tau_t

# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--pairs", type=int, default=4000)
    # DEFAULT per OMZ IR (come verificato nelle tue run "buone"):
    ap.add_argument("--swaprb", type=int, choices=[0, 1], default=0, help="0=BGR, 1=RGB")
    ap.add_argument("--preproc", type=str, choices=["raw255", "arcface", "imagenet"], default="raw255")
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    compiled, layout, inp_port, out_port, out_kind = load_ov_session(Path(args.xml))
    swaprb = bool(args.swaprb)
    preproc = args.preproc

    print(f"[cfg] layout={layout} swaprb={swaprb} preproc={preproc}")

    X, y, _, _ = load_split(compiled, layout, inp_port, out_port, out_kind,
                        args.split, swaprb=swaprb)

    # sanity prints utili
    if X.size:
        print(f"[emb] std(mean over dims) = {float(np.std(X, axis=0).mean()):.6f}")
    binc = np.bincount(y) if y.size else np.array([])
    if binc.size:
        print(f"[data] n_id={len(binc)} imgs/id min/mean/max = {int(binc.min())}/{float(binc.mean()):.1f}/{int(binc.max())}")

    pos, neg = build_pairs_safe(y, max_pos=args.pairs, max_neg=args.pairs)
    print(f"[INFO] Coppie generate -> positive: {len(pos)} | negative: {len(neg)}")
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Servono >=2 immagini della stessa identità e >=2 identità diverse.")

    s_pos = cosine(X[pos[:,0]], X[pos[:,1]])
    s_neg = cosine(X[neg[:,0]], X[neg[:,1]])
    print(f"cos(pos): mean={s_pos.mean():.3f}, min={s_pos.min():.3f}, max={s_pos.max():.3f}")
    print(f"cos(neg): mean={s_neg.mean():.3f}, min={s_neg.min():.3f}, max={s_neg.max():.3f}")

    y_true = np.r_[np.ones_like(s_pos), np.zeros_like(s_neg)]
    y_score = np.r_[s_pos, s_neg]

    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)

    # soglia all’EER (punto |1 - tpr - fpr| minimo)
    idx = np.nanargmin(np.abs(1 - tpr - fpr))
    eer = (fpr[idx] + (1 - tpr[idx]))/2
    tau_eer = thr[idx]

    # TPR @ FPR target con interpolazione
    tpr_1e3, tau_1e3 = tpr_tau_at_fpr_interp(fpr, tpr, thr, 1e-3)
    tpr_1e4, tau_1e4 = tpr_tau_at_fpr_interp(fpr, tpr, thr, 1e-4)

    print(f"AUC={auc_val:.4f} | EER≈{eer:.4f}")
    print(f"Soglia τ @EER: {tau_eer:.4f}")
    print(f"TPR@FPR=1e-3: {tpr_1e3:.3f} (τ≈{tau_1e3:.4f})")
    print(f"TPR@FPR=1e-4: {tpr_1e4:.3f} (τ≈{tau_1e4:.4f})")

if __name__ == "__main__":
    main()
