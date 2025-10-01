# detect_crop_robust.py
import cv2, sys
from pathlib import Path
import numpy as np

def rotate_bound(image, angle): # ruota senza tagliare
    (h, w) = image.shape[:2] # altezza, larghezza
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0) # centro, angolo, scala
    cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1]) # calcola nuove dimensioni
    nW = int((h * sin) + (w * cos)) # nuova larghezza
    nH = int((h * cos) + (w * sin)) # nuova altezza
    M[0, 2] += (nW / 2) - w/2 # trasla
    M[1, 2] += (nH / 2) - h/2 # trasla
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR) # rotazione

def detect_once(gray, cascade, params): # prova più set di parametri
    for (sf, mn, ms) in params: # scaleFactor, minNeighbors, minSize
        faces = cascade.detectMultiScale(  
            gray, scaleFactor=sf, minNeighbors=mn, minSize=ms
        ) # faces = [(x,y,w,h), ...]
         # se trova, ritorna subito
        if len(faces): 
            return faces
    return ()

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None 
    if not img_path: 
        print("Usage: python detect_crop_robust.py <image_path>")
        sys.exit(1)

    p = Path(img_path).expanduser()
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"Image not found: {p}")

    # Preproc: grayscale + CLAHE (aiuta tanto Haar) -> CLAHE cos'è?  https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    gray_eq = clahe.apply(gray) 

    # Carica cascades
    face_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    assert not face_frontal.empty(), "Frontal cascade non caricato"
    assert not face_profile.empty(), "Profile cascade non caricato"

    # Parametri “sensibili” → più permissivi in fallback
    param_sets = [
        (1.1, 5, (80,80)),    # come prima
        (1.08, 4, (60,60)),
        (1.05, 3, (40,40)),
        (1.03, 3, (30,30))
    ]
    angles = [0, -15, 15, -25, 25, -35, 35]

    best = None  # (area, angle, (x,y,w,h), rotated_img)
    for ang in angles:
        g = gray_eq if ang == 0 else cv2.cvtColor(rotate_bound(img, ang), cv2.COLOR_BGR2GRAY)
        if ang != 0:
            g = clahe.apply(g)

        # Frontal first
        faces = detect_once(g, face_frontal, param_sets)
        # Fallback: profilo (sia normale che specchiato)
        if len(faces) == 0:
            faces = detect_once(g, face_profile, param_sets)
            if len(faces) == 0:
                g_flip = cv2.flip(g, 1)
                faces = detect_once(g_flip, face_profile, param_sets)

        for (x,y,w,h) in faces:
            area = w*h
            if best is None or area > best[0]:
                # ricostruisci la BGR ruotata per cropping finale
                rotated = img if ang == 0 else rotate_bound(img, ang)
                best = (area, ang, (x,y,w,h), rotated)

    out_dir = Path("out"); out_dir.mkdir(exist_ok=True)

    if best is None:
        print("Nessun volto trovato anche con rotazioni/fallback.")
        cv2.imwrite(str(out_dir/"detected.png"), img)
        return

    _, ang, (x,y,w,h), im_rot = best
    face = im_rot[y:y+h, x:x+w]
    face = cv2.resize(face, (160,160))
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(20,20))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        (lx,ly, lw, lh), (rx,ry, rw, rh) = sorted(eyes, key=lambda e: e[0])  # sinistra/destra
        cx_l, cy_l = lx+lw/2, ly+lh/2
        cx_r, cy_r = rx+rw/2, ry+rh/2
        angle = np.degrees(np.arctan2(cy_r - cy_l, cx_r - cx_l))
        M = cv2.getRotationMatrix2D((face.shape[1]/2, face.shape[0]/2), angle, 1.0)
        face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]), flags=cv2.INTER_LINEAR)
        # crop centrale per tagliare eventuali bordi neri
        pad = int(0.08 * min(face.shape[:2]))
        face = face[pad:face.shape[0]-pad, pad:face.shape[1]-pad]

    face = cv2.resize(face, (160,160))
    cv2.imwrite(str(out_dir/"crop_0.png"), face)
    
    # Disegna bbox per debug
    dbg = im_rot.copy()
    cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(dbg, f"angle={ang}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imwrite(str(out_dir/"detected.png"), dbg)
    cv2.imwrite(str(out_dir/"crop_0.png"), face)
    print(f"​✅​: volto trovato a angle={ang}°, salvati out/detected.png e out/crop_0.png")

if __name__ == "__main__":
    main()
