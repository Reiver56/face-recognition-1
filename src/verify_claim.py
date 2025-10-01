# src/verify_claim.py
import cv2, argparse, json
import numpy as np
from pathlib import Path
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "aligned"
MODEL_DIR = ROOT / "models"

def load_onnx_session(onnx_path: Path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    layout = "NCHW"
    if len(shape) == 4:
        if isinstance(shape[1], int) and shape[1] not in (1,3):
            if isinstance(shape[-1], int) and shape[-1] in (1,3):
                layout = "NHWC"
        elif shape[1] is None and shape[-1] in (1,3):
            layout = "NHWC"
    return sess, layout, inp.name

def onnx_embed(sess, layout, input_name, face_bgr):
    img = cv2.resize(face_bgr, (112,112))
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0/127.5, size=(112,112),
        mean=(127.5,127.5,127.5), swapRB=True, crop=False
    ).astype(np.float32)  # (1,3,112,112)
    x = blob if layout=="NCHW" else np.transpose(blob, (0,2,3,1))
    out = sess.run(None, {input_name: x})[0]
    feat = out.reshape(-1).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-12)
    return feat

def build_templates(sess, layout, input_name, split="train"):
    T = {}
    root = ALIGNED / split
    ids = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    for pid in ids:
        feats = []
        for imgp in sorted((root/pid).glob("*.*")):
            img = cv2.imread(str(imgp))
            if img is None: 
                continue
            feats.append(onnx_embed(sess, layout, input_name, img))
        if feats:
            T[pid] = np.vstack(feats)
    if not T:
        raise RuntimeError("Nessun template costruito.")
    return T

def cosine_max(emb, mat):
    return float((mat @ emb).max())

def verify_claim(img_path, claimed_id, sess, layout, input_name, templates, tau=0.35, margin=0.0, topk=3):
    img = cv2.imread(str(img_path))
    if img is None:
        return False, {"error": f"Immagine non leggibile: {img_path}"}
    emb = onnx_embed(sess, layout, input_name, img)

    if claimed_id not in templates:
        return False, {"error": f"Identità '{claimed_id}' non presente."}

    s_pos = cosine_max(emb, templates[claimed_id])

    impostor_scores = []
    for pid, feats in templates.items():
        if pid == claimed_id: 
            continue
        impostor_scores.append((pid, cosine_max(emb, feats)))
    impostor_scores.sort(key=lambda x: x[1], reverse=True)
    best_imp_id, best_imp = (impostor_scores[0] if impostor_scores else ("", -1.0))

    accept = (s_pos >= tau) and (s_pos - best_imp >= margin)

    info = {
        "claimed_id": claimed_id,
        "score_claimed": s_pos,
        "best_impostor_id": best_imp_id,
        "best_impostor_score": best_imp,
        "tau": tau,
        "margin": margin,
        "accepted": bool(accept),
        "top_impostors": impostor_scores[:topk],
    }
    return accept, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default=str(MODEL_DIR/"arcface_r100.onnx"))
    ap.add_argument("--img", required=True)
    ap.add_argument("--id",  required=True)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    sess, layout, input_name = load_onnx_session(Path(args.onnx))
    templates = build_templates(sess, layout, input_name, split="train")
    ok, info = verify_claim(args.img, args.id, sess, layout, input_name, templates, tau=args.tau, margin=args.margin)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("\nDECISIONE:", "OK, SEI TU ✅" if ok else "NO, NON SEI TU ❌")

if __name__ == "__main__":
    main()
