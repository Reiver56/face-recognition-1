import argparse, cv2, numpy as np
from pathlib import Path
from openvino.runtime import Core

SRC_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)

def affine_by_5pts(pts: np.ndarray, image: np.ndarray, out_size=(112,112)):
    assert pts.shape==(5,2)
    M,_ = cv2.estimateAffinePartial2D(pts.astype(np.float32), SRC_5PTS, method=cv2.LMEDS)
    return cv2.warpAffine(image, M, out_size, flags=cv2.INTER_LINEAR, borderValue=0)

def largest_box(dets, w, h):
    # dets: [N,7] => [img_id, label, conf, xmin, ymin, xmax, ymax] (norm.)
    if dets is None or dets.size==0: return None
    dets = dets[dets[:,2] > 0.6]  # conf
    if not len(dets): return None
    # to absolute
    boxes = dets[:,3:7].copy()
    boxes[:,[0,2]] *= w
    boxes[:,[1,3]] *= h
    # pick largest area
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    i = int(np.argmax(areas))
    x1,y1,x2,y2 = boxes[i]
    x1,y1,x2,y2 = map(int, [max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)])
    return x1,y1,x2,y2

def five_points_from_reg(output, roi):
    # landmarks-regression-retail-0009: output shape (1,10) in [0..1] sul ROI
    lmk = output.reshape(-1)  # 10 vals
    pts = np.stack([lmk[0::2], lmk[1::2]], axis=1)  # (5,2) norm
    x1,y1,x2,y2 = roi
    rw, rh = (x2-x1), (y2-y1)
    pts[:,0] = x1 + pts[:,0]*rw
    pts[:,1] = y1 + pts[:,1]*rh
    # Riordino robusto a occhi->naso->bocca usando la Y:
    order = np.argsort(pts[:,1])  # su
    eyes = pts[order[:2]]; mouth = pts[order[-2:]]; nose = pts[order[2:3]][0]
    # left/right by x
    le,re = (eyes[0], eyes[1]) if eyes[0,0] < eyes[1,0] else (eyes[1], eyes[0])
    lm,rm = (mouth[0], mouth[1]) if mouth[0,0] < mouth[1,0] else (mouth[1], mouth[0])
    return np.vstack([le, re, nose, lm, rm])

def build_ov_models(det_xml, lmk_xml, device="CPU"):
    core = Core()
    det = core.compile_model(core.read_model(det_xml), device)
    lmk = core.compile_model(core.read_model(lmk_xml), device)
    return det, lmk

def detect(core_model, img):
    # face-detection-retail-0004: input 1x3x300x300, output 1x1xNx7
    inp = core_model.input(0)
    H = inp.shape[2]; W = inp.shape[3]
    blob = cv2.dnn.blobFromImage(img, size=(W,H), swapRB=False, crop=False)  # raw 0..255
    res = core_model.create_infer_request().infer({inp: blob})
    out = list(res.values())[0]  # (1,1,N,7)
    return out.reshape(-1,7)

def landmarks(core_model, roi_img):
    # landmarks-regression-retail-0009: input 1x3x48x48, output 1x10
    inp = core_model.input(0)
    H = inp.shape[2]; W = inp.shape[3]
    roi_resized = cv2.resize(roi_img, (W,H))
    blob = cv2.dnn.blobFromImage(roi_resized, size=(W,H), swapRB=False, crop=False)
    res = core_model.create_infer_request().infer({inp: blob})
    out = list(res.values())[0]
    return out

def process_image(det, lmk, img_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None: return False
    h,w = img.shape[:2]
    dets = detect(det, img)
    box = largest_box(dets, w, h)
    if box is None:
        return False
    x1,y1,x2,y2 = box
    roi = img[y1:y2, x1:x2]
    out = landmarks(lmk, roi)
    pts5 = five_points_from_reg(out, (x1,y1,x2,y2))
    aligned = affine_by_5pts(pts5, img, (112,112))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), aligned)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-xml", required=True)
    ap.add_argument("--lmk-xml", required=True)
    ap.add_argument("--src", required=True, help="root raw (es. data/raw/lfw)")
    ap.add_argument("--dst", required=True, help="root aligned (es. data/aligned/lfw)")
    args = ap.parse_args()

    det, lmk = build_ov_models(args.det_xml, args.lmk_xml)

    src = Path(args.src); dst = Path(args.dst)
    persons = sorted([d for d in src.iterdir() if d.is_dir()])
    total, ok = 0, 0
    for person in persons:
        for imgp in sorted(person.glob("*.jpg")):
            total += 1
            rel = imgp.relative_to(src)
            outp = dst / rel
            ok += process_image(det, lmk, imgp, outp)
            if total % 100 == 0:
                print(f"[{ok}/{total}] {ok/total:.1%}")
    print(f"[DONE] aligned {ok}/{total} ({ok/total:.1%}) -> {dst}")

if __name__ == "__main__":
    main()
