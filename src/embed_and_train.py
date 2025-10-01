import cv2, argparse, json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from joblib import dump
<<<<<<< HEAD
import onnxruntime as ort
=======
>>>>>>> f22d42292ff0db868e0f3e556f91d39943215173

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "aligned"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

<<<<<<< HEAD
def load_onnx_session(onnx_path: Path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape  # es. [1,3,112,112] o [1,112,112,3] o [None,3,112,112]
    # prova a dedurre il layout
    layout = "NCHW"
    if len(shape) == 4:
        # se la seconda dim è 3 (o 1), è quasi certamente NCHW
        if isinstance(shape[1], int) and shape[1] not in (1,3):
            # magari è NHWC
            if isinstance(shape[-1], int) and shape[-1] in (1,3):
                layout = "NHWC"
        elif shape[1] is None and shape[-1] in (1,3):
            layout = "NHWC"
    return sess, layout, inp.name

def preproc_params(onnx_path: Path):
    name = str(onnx_path).lower()
    if "sface" in name or "face_recognition_sface" in name:
        # SFace (OpenCV Zoo): BGR, niente swapRB
        return dict(size=(112,112), scale=1.0/128.0, mean=(127.5,127.5,127.5), swapRB=False)
    else:
        # ArcFace / InsightFace: RGB, swapRB=True
        return dict(size=(112,112), scale=1.0/127.5, mean=(127.5,127.5,127.5), swapRB=True)
    
def onnx_embed(sess, layout, input_name, face_bgr, pp):
    img = cv2.resize(face_bgr, pp["size"])
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=pp["scale"], size=pp["size"],
        mean=pp["mean"], swapRB=pp["swapRB"], crop=False
    ).astype(np.float32)  # (1,3,112,112) NCHW

    x = blob if layout == "NCHW" else np.transpose(blob, (0,2,3,1))
    out = sess.run(None, {input_name: x})[0]
    feat = out.reshape(-1).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-12)
    return feat

def load_split_feats(sess, layout, input_name, pp, split="train"):
=======
def load_arcface(model_path: Path):
    net = cv2.dnn.readNet(str(model_path))
    return net

def arcface_embed(net, face_bgr):
    # Preprocess ArcFace: 112x112, RGB, mean/std 127.5
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112))
    blob = cv2.dnn.blobFromImage(
        rgb, 1.0/127.5, (112,112), (127.5,127.5,127.5),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    feat = net.forward().reshape(-1).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-12)  # L2 normalize
    return feat

def load_split_feats(net, split="train"):
>>>>>>> f22d42292ff0db868e0f3e556f91d39943215173
    root = ALIGNED / split
    classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    X, y, labels = [], [], []
    for lid, cname in enumerate(classes):
        labels.append(cname)
        for imgp in sorted((root/cname).glob("*.*")):
            img = cv2.imread(str(imgp))
            if img is None:
                continue
<<<<<<< HEAD
            X.append(onnx_embed(sess, layout, input_name, img, pp))
=======
            X.append(arcface_embed(net, img))
>>>>>>> f22d42292ff0db868e0f3e556f91d39943215173
            y.append(lid)
    if not X:
        raise RuntimeError(f"Nessuna immagine trovata in {root}. Hai eseguito 01_detect_align_landmarks.py?")
    X = normalize(np.vstack(X))
    y = np.array(y, dtype=np.int32)
    return X, y, labels

def main():
    ap = argparse.ArgumentParser()
<<<<<<< HEAD
    ap.add_argument("--onnx", type=str, default=str(MODEL_DIR/"arcface_r100.onnx"),
                    help="Percorso al modello ONNX (es. arcface_r100.onnx o sface.onnx)")
=======
    ap.add_argument("--arcface", type=str, default=str(MODEL_DIR/"arcface_r100.onnx"))
>>>>>>> f22d42292ff0db868e0f3e556f91d39943215173
    ap.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf"])
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

<<<<<<< HEAD
    onnx_path = Path(args.onnx)
    sess, layout, input_name = load_onnx_session(onnx_path)
    pp = preproc_params(onnx_path)
    print(f"[INFO] ONNX caricato. Layout input: {layout} | swapRB={pp['swapRB']} | scale={pp['scale']}")

    Xtr, ytr, classes = load_split_feats(sess, layout, input_name, pp, "train")
=======
    net = load_arcface(Path(args.arcface))
    Xtr, ytr, classes = load_split_feats(net, "train")

>>>>>>> f22d42292ff0db868e0f3e556f91d39943215173
    clf = SVC(kernel=args.kernel, C=args.C, probability=True)
    clf.fit(Xtr, ytr)

    dump(clf, MODEL_DIR/"svm.pkl")
    with open(MODEL_DIR/"classes.json", "w") as f:
        json.dump(classes, f)

    print(f"Addestrate {len(classes)} classi su {len(Xtr)} campioni.")
    print("Salvati: models/svm.pkl e models/classes.json")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> f22d42292ff0db868e0f3e556f91d39943215173
