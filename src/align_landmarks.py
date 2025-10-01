import cv2, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" # cartella con immagini non allineate
ALIGNED = ROOT / "data" / "aligned"

# Template 5 punti (ArcFace, 112x112). Verrà scalato a --size.
ARC_TEMPLATE_112 = np.float32([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
])

# Indici del modello 68 punti (dlib-style) usato da LBF:
IDX_LEFT_EYE  = list(range(36, 42))
IDX_RIGHT_EYE = list(range(42, 48))
IDX_NOSE_TIP  = [30]
IDX_MOUTH_LR  = [48, 54]

def eye_center(pts, idxs):
    sel = pts[idxs]
    return sel.mean(axis=0)

def extract_5pts(land68):
    """land68: (68,2) → 5 punti (LE, RE, nose, mouthL, mouthR)"""
    le = eye_center(land68, IDX_LEFT_EYE)
    re = eye_center(land68, IDX_RIGHT_EYE)
    nose = land68[IDX_NOSE_TIP][0]
    ml = land68[IDX_MOUTH_LR][0]
    mr = land68[IDX_MOUTH_LR][1]
    return np.float32([le, re, nose, ml, mr])

def align_face(img_bgr, land68, out_size=160):
    src5 = extract_5pts(land68)
    scale = out_size / 112.0
    dst5 = ARC_TEMPLATE_112 * scale
    # stima trasformazione di similitudine (rot+scale+transl)
    M, _ = cv2.estimateAffinePartial2D(src5, dst5, method=cv2.LMEDS)
    if M is None:
        # fallback: usa solo occhi → rotazione semplice
        le, re = src5[0], src5[1]
        angle = np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0]))
        center = tuple(np.mean([le, re], axis=0))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        warped = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]),
                                flags=cv2.INTER_LINEAR)
        # recrop attorno al bounding box del volto? qui semplifichiamo: ridimensiona
        return cv2.resize(warped, (out_size, out_size))
    return cv2.warpAffine(img_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR)

def process_split(split="train", out_size=160, minsize=60, lbf_path="models/lbfmodel.yaml"):
    in_dir  = RAW / split
    out_dir = ALIGNED / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detector volto (Haar) + Facemark LBF
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(lbf_path)

    saved, total = 0, 0
    for person in sorted(in_dir.iterdir()):
        if not person.is_dir(): continue
        (out_dir / person.name).mkdir(parents=True, exist_ok=True)
        for imgp in sorted(person.glob("*.*")):
            total += 1
            img = cv2.imread(str(imgp))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(minsize, minsize))
            if len(faces) == 0:
                continue

            # prendi il volto più grande
            x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
            roi = img[y:y+h, x:x+w]
            gray_roi = gray[y:y+h, x:x+w]
            # facemark vuole rect in coordinate ROI
            rects = np.array([[0, 0, w, h]])

            ok, landmarks = facemark.fit(gray_roi, rects)
            if not ok or len(landmarks)==0:
                continue
            land68 = landmarks[0][0]  # shape (68,2) relativo alla ROI
            # porta i landmarks in coordinate immagine
            land68[:,0] += x; land68[:,1] += y

            aligned = align_face(img, land68, out_size=out_size)
            out_path = out_dir / person.name / f"{imgp.stem}.png"
            cv2.imwrite(str(out_path), aligned)
            saved += 1

    print(f"[{split}] salvati {saved}/{total} crop allineati in {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train","test"])
    ap.add_argument("--size", type=int, default=160, help="dimensione output (consiglio 160; l'embedder ridimensiona a 112)")
    ap.add_argument("--minsize", type=int, default=60)
    ap.add_argument("--lbf", type=str, default="models/lbfmodel.yaml")
    args = ap.parse_args()
    process_split(args.split, args.size, args.minsize, args.lbf)
