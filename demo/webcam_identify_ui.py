#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys, time, os, atexit, tempfile
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
from openvino.runtime import Core

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from arcface_ov import ArcFaceOV

# ------------------ utils GUI ------------------
def has_gui() -> bool:
    try:
        cv2.namedWindow("__t__"); cv2.destroyWindow("__t__")
        return True
    except Exception:
        return False

def draw_text(img, txt, org, scale=0.5, color=(0,0,0), thickness=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def mk_panel(w, h, bg=(245,245,245)):
    panel = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.rectangle(panel, (0,0), (w-1,h-1), (220,220,220), 1)
    return panel

def draw_pipeline(panel, stages, x0=18, y0=20, dx=22, margin=12):
    H, W = panel.shape[:2]
    draw_text(panel, "PIPELINE", (x0, max(0, y0-6)), 0.6, (60,60,60), 2)

    usable_w = max(80, W - 2*margin)
    n = max(1, len(stages))
    min_box_w, max_box_w = 120, 200
    box_w = max(min_box_w, min(max_box_w, (usable_w - (n-1)*dx) // n))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    pad = 8
    one_line_h = 36
    two_line_h = 54

    x, y = x0, y0
    last_row_h = one_line_h

    for i, st in enumerate(stages):
        name_txt = st["name"]
        ms_txt   = f'{st["ms"]:.1f} ms'

        name_w, _ = cv2.getTextSize(name_txt, font, scale, 2)[0]
        ms_w,   _ = cv2.getTextSize(ms_txt,   font, scale, 2)[0]
        two_lines = (pad + name_w + pad + ms_w + pad) > (box_w - 2)
        box_h = two_line_h if two_lines else one_line_h

        need = box_w + (dx + 10 if i < n-1 else 0)
        if x + need + margin > W:
            x = x0
            y += last_row_h + 16
        last_row_h = box_h

        x1, y1 = x, y
        x2 = min(x + box_w, W - margin)
        y2 = min(y + box_h, H - margin)

        base_col = (60,180,75) if st.get("ok", True) else (30,105,210)
        col = st.get("color", base_col)
        cv2.rectangle(panel, (x1, y1), (x2, y2), col, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (255,255,255), 2)

        if two_lines:
            draw_text(panel, name_txt, (x1 + pad, min(y1 + 18, H - 6)), scale, (255,255,255), 2)
            draw_text(panel, ms_txt,   (x1 + pad, min(y1 + 34, H - 6)), scale, (255,255,255), 2)
        else:
            ty = min(y1 + 22, H - 6)
            draw_text(panel, name_txt, (x1 + pad, ty), scale, (255,255,255), 2)
            draw_text(panel, ms_txt,   (x2 - pad - ms_w, ty), scale, (255,255,255), 2)

        if i < n - 1:
            ymid = y1 + (y2 - y1) // 2
            ax1 = min(x2, W - margin - 1)
            ax2 = min(x2 + dx, W - margin - 1)
            if ax2 > ax1 and 0 <= ymid < H:
                cv2.arrowedLine(panel, (ax1, ymid), (ax2, ymid), (120,120,120), 2, tipLength=0.4)

        x += box_w + dx

    return y + last_row_h + 20

def draw_topk(panel, top, thumbs, start_y=110, row_h=74, thumb_wh=(68,68), tau=0.30):
    H, W = panel.shape[:2]
    draw_text(panel, f"TOP-K  (τ={tau:.2f})", (18, start_y-10), 0.6, (60,60,60), 2)
    max_rows = max(0, (H - start_y) // row_h)
    items = top[:max_rows] if max_rows > 0 else []
    y = start_y
    for rank, (score, label, path) in enumerate(items, 1):
        thumb = thumbs.get(path)
        if thumb is None:
            try:
                im = cv2.imread(path)
                if im is None: raise FileNotFoundError
                thumb = cv2.resize(im, thumb_wh)
            except Exception:
                thumb = np.full((thumb_wh[1], thumb_wh[0], 3), (200,200,200), np.uint8)
            thumbs[path] = thumb

        y0 = y - 4
        y1 = min(y + row_h - 10, H)
        if y0 >= H: break
        cv2.rectangle(panel, (12, max(0, y0)), (W-12, y1), (235,235,235), -1)
        cv2.rectangle(panel, (12, max(0, y0)), (W-12, y1), (210,210,210), 1)

        th, tw = thumb_wh[1], thumb_wh[0]
        y_t0 = y; y_t1 = min(y + th, H)
        if y_t1 > y_t0:
            crop_h = y_t1 - y_t0
            panel[y_t0:y_t1, 18:18+tw] = thumb[:crop_h, :, :]

        name = str(label)
        draw_text(panel, f"{rank}. {name[:22]}", (18+tw+10, min(y+20, H-6)), 0.6, (20,20,20), 2)
        draw_text(panel, f"s={score:.3f}", (18+tw+10, min(y+44, H-6)), 0.6,
                  (0,140,0) if score>=tau else (0,0,200), 2)
        y += row_h

def imshow_fit(window_name: str, img: np.ndarray):
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        if win_w > 0 and win_h > 0:
            ih, iw = img.shape[:2]
            scale = min(win_w / max(1, iw), win_h / max(1, ih))
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            vis = cv2.resize(img, (new_w, new_h))
            canvas = np.zeros((win_h, win_w, 3), dtype=img.dtype)
            x = (win_w - new_w) // 2
            y = (win_h - new_h) // 2
            canvas[y:y+new_h, x:x+new_w] = vis
            cv2.imshow(window_name, canvas); return
    except Exception:
        pass
    cv2.imshow(window_name, img)

# ------------------ OpenVINO wrappers ------------------
class OVModel:
    def __init__(self, xml_path: str):
        core = Core()
        model = core.read_model(xml_path)
        self.exec = core.compile_model(model, "CPU")
        self.inp = self.exec.input(0)
        self.outs = [self.exec.output(i) for i in range(len(self.exec.outputs))]
        self.shape = [int(d.get_length()) if d.is_static else -1
                      for d in model.inputs[0].partial_shape]

def detect_fd_ret04(det: OVModel, frame_bgr: np.ndarray, conf_thr=0.6):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, size=(300,300), swapRB=False)
    res = det.exec.create_infer_request().infer({det.inp: blob})
    out = res[det.outs[0]].squeeze(axis=0)
    dets = out[0] if out.ndim == 3 else out
    boxes = []
    for d in dets:
        if d[2] < conf_thr: continue
        x1 = int(d[3]*w); y1 = int(d[4]*h)
        x2 = int(d[5]*w); y2 = int(d[6]*h)
        x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
        if x2 > x1 and y2 > y1:
            boxes.append((x1,y1,x2,y2,float(d[2])))
    return boxes

def landmarks_lmr0009(lmk: OVModel, frame_bgr: np.ndarray, box):
    x1,y1,x2,y2,_ = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0: return None
    blob = cv2.dnn.blobFromImage(crop, size=(48,48), swapRB=False)
    res = lmk.exec.create_infer_request().infer({lmk.inp: blob})
    pts = res[lmk.outs[0]].reshape(5,2).astype(np.float32)
    pts[:,0] = x1 + pts[:,0]*(x2-x1)
    pts[:,1] = y1 + pts[:,1]*(y2-y1)
    return pts

# ------------------ Align + cosine ------------------
ARC_TARGET_5P = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_by_5p(img_bgr: np.ndarray, pts5_abs: np.ndarray, size=(112,112)):
    if pts5_abs is None or len(pts5_abs) != 5:
        return cv2.resize(img_bgr, size)
    M, _ = cv2.estimateAffinePartial2D(pts5_abs, ARC_TARGET_5P, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img_bgr, size)
    return cv2.warpAffine(img_bgr, M, size, flags=cv2.INTER_LINEAR)

def cosine_matrix(q: np.ndarray, E: np.ndarray) -> np.ndarray:
    q = q.reshape(1, -1).astype(np.float32)
    return (E @ q.T).ravel() / ((np.linalg.norm(E,axis=1)*np.linalg.norm(q))+1e-12)

# ------------------ capture helpers ------------------
def parse_source(src_str: str):
    return int(src_str) if src_str.isdigit() else src_str

def open_cam(src):
    tried = []
    def _try(open_args, tag):
        cap = cv2.VideoCapture(*open_args)
        ok = cap.isOpened()
        if ok: ok, _ = cap.read()
        if not ok: cap.release()
        tried.append(f"{tag}: {'ok' if ok else 'no frames'}")
        return cap if ok else None

    cap = _try((src,), "default")
    if cap: return cap
    if hasattr(cv2, "CAP_MSMF"):
        cap = _try((src, cv2.CAP_MSMF), "msmf")
        if cap: return cap
    if hasattr(cv2, "CAP_DSHOW"):
        cap = _try((src, cv2.CAP_DSHOW), "dshow")
        if cap: return cap

    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, _ = cap.read()
        tried.append(f"mjpg/640x480: {'ok' if ok else 'no frames'}")
        if ok: return cap
        cap.release()
    raise RuntimeError(f"Impossibile aprire la sorgente '{src}'. Tentativi: {tried}")

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--det-xml", required=True)
    ap.add_argument("--lmk-xml", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--right", type=int, default=380)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--source", default="0")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--search-warn-n", type=int, default=10000)
    ap.add_argument("--enroll-dir", default="")
    ap.add_argument("--enroll-name", default="Me")
    ap.add_argument("--index-save", default="")
    ap.add_argument("--no-autosave", action="store_true",
                    help="non salvare automaticamente su uscita")
    args = ap.parse_args()

    args.no_gui = args.no_gui or not has_gui()
    src = parse_source(args.source)

    cfg = json.loads(Path(args.config).read_text())
    tau = args.tau if args.tau is not None else float(cfg["thresholds"]["default"])

    arc = ArcFaceOV(cfg["xml"], swaprb=cfg.get("swaprb", False), preproc=cfg.get("preproc","raw255"))
    det = OVModel(args.det_xml)
    lmk_model = None if args.no_align else OVModel(args.lmk_xml)
    align_enabled = (not args.no_align)

    # --- index loading with manifest support and robust fallbacks ---
    user_index = Path(args.index)
    manifest = user_index.parent / "latest_index.json"
    index_to_load = user_index
    loaded = None

    if manifest.exists():
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            candidate = user_index.parent / m.get("last", "")
            if candidate.exists():
                try:
                    loaded = np.load(str(candidate), allow_pickle=True)
                    index_to_load = candidate
                    print(f"[info] found latest index -> using {candidate} (entries={m.get('entries','?')})")
                except Exception as e:
                    print(f"[warn] manifest points to {candidate} but loading failed: {e}; falling back to {user_index}")
        except Exception as e:
            print(f"[warn] could not parse manifest {manifest}: {e}")

    if loaded is None:
        if not user_index.exists():
            raise RuntimeError(f"Index file {user_index} not found and no manifest present.")
        try:
            loaded = np.load(str(user_index), allow_pickle=True)
        except Exception as e:
            print(f"[warn] failed to load index '{user_index}': {e}")
            # scan other npz in same dir
            cand_files = sorted(list(user_index.parent.glob("*.npz")), key=lambda p: p.stat().st_mtime, reverse=True)
            for c in cand_files:
                if c.resolve() == user_index.resolve(): continue
                try:
                    print(f"[info] trying alternative index {c}")
                    loaded = np.load(str(c), allow_pickle=True)
                    index_to_load = c
                    break
                except Exception as e2:
                    print(f"[warn] can't load {c}: {e2}")
            if loaded is None:
                raise RuntimeError(f"Unable to load any valid index in {user_index.parent}. Please re-create index.")

    data = loaded
    E = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(object)
    paths = data["paths"].astype(object)

    # save_path (if --index-save provided use it, else overwrite loaded file)
    save_path = Path(args.index_save) if args.index_save else Path(index_to_load)
    gallery_dirty = False

    def do_save():
        """Salva l'indice in modo robusto (tmp + os.replace) e aggiorna latest_index.json."""
        nonlocal gallery_dirty, E, labels, paths, save_path
        if not gallery_dirty:
            print("[save] nulla da salvare.")
            return
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # tmp nello stesso folder per atomic replace
        fd, tmp_path = tempfile.mkstemp(prefix=save_path.stem + "_", suffix=save_path.suffix + ".tmp",
                                        dir=str(save_path.parent))
        os.close(fd)
        try:
            np.savez_compressed(str(tmp_path), embeddings=E, labels=labels, paths=paths)
            if os.path.getsize(tmp_path) == 0:
                raise RuntimeError("file temporaneo vuoto, salvataggio fallito")
            os.replace(tmp_path, save_path)
            print(f"[save] index aggiornato -> {save_path} (entries={E.shape[0]})")
            gallery_dirty = False
            # manifest
            manifest_file = save_path.parent / "latest_index.json"
            mfd, mtmp = tempfile.mkstemp(prefix="latest_index_", suffix=".tmp", dir=str(save_path.parent))
            os.close(mfd)
            meta = {
                "last": save_path.name,
                "entries": int(E.shape[0]),
                "saved_at": datetime.now(timezone.utc).isoformat()
            }
            with open(mtmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
                fh.flush(); os.fsync(fh.fileno())
            os.replace(mtmp, manifest_file)
        except Exception as e:
            print(f"[save][ERROR] fallito: {e}")
            try:
                if os.path.exists(tmp_path): os.unlink(tmp_path)
            except Exception:
                pass

    if not args.no_autosave:
        atexit.register(do_save)

    cap = open_cam(src)

    thumbs = {}
    save_dir = Path(args.save_dir) if (args.no_gui and args.save_dir) else None
    if save_dir: save_dir.mkdir(parents=True, exist_ok=True)
    enroll_dir = Path(args.enroll_dir) if args.enroll_dir else None
    if enroll_dir: enroll_dir.mkdir(parents=True, exist_ok=True)

    WIN = "ArcFace (OpenVINO) — webcam + pipeline"
    window_created = False

    if not args.no_gui:
        print("[info] q=quit | a=align on/off | +/-=threshold | p=screenshot | e=enroll | d=delete | s=save | n=rename enroll")

    fps_hist = deque(maxlen=30)
    frame_id = 0
    best_crop = best_emb = best_top = None

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("[warn] frame non disponibile, stop.")
            break

        H, W = frame.shape[:2]
        right_w = args.right

        if (not args.no_gui) and (not window_created):
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, W + right_w, H)
            except Exception:
                pass
            window_created = True

        panel = mk_panel(right_w, H)

        # Detect
        td0 = time.perf_counter()
        boxes = detect_fd_ret04(det, frame, conf_thr=0.6)
        td = (time.perf_counter()-td0)*1000

        t_lmk = t_align = t_emb = t_srch = 0.0
        best_face_for_panel = None
        best_face_score = -1.0
        headless_lines = []

        best_crop = best_emb = best_top = None

        for (x1,y1,x2,y2,conf) in boxes:
            tl0 = time.perf_counter()
            pts5 = landmarks_lmr0009(lmk_model, frame, (x1,y1,x2,y2,conf)) if (align_enabled and lmk_model) else None
            t_lmk += (time.perf_counter()-tl0)*1000

            ta0 = time.perf_counter()
            crop = align_by_5p(frame, pts5, size=(112,112)) if (align_enabled and pts5 is not None) \
                   else cv2.resize(frame[y1:y2, x1:x2], (112,112))
            t_align += (time.perf_counter()-ta0)*1000

            te0 = time.perf_counter()
            f = arc.embed_bgr(crop)
            t_emb += (time.perf_counter()-te0)*1000

            ts0 = time.perf_counter()
            s = cosine_matrix(f, E)
            idx = int(np.argmax(s)); smax = float(s[idx]); lab = str(labels[idx])
            order = np.argsort(-s)[:args.topk]
            top = [(float(s[i]), str(labels[i]), str(paths[i])) for i in order]
            t_srch += (time.perf_counter()-ts0)*1000

            name = lab if smax >= tau else "UNKNOWN"
            color = (0,200,0) if name != "UNKNOWN" else (0,0,200)

            if not args.no_gui:
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{name} {smax:.2f}", (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            headless_lines.append(f"[face] box=({x1},{y1},{x2},{y2}) name={name} score={smax:.3f}")

            if smax > best_face_score:
                best_face_score = smax
                best_face_for_panel = top
                best_crop = crop; best_emb = f; best_top = top

        # Pipeline panel
        if not args.no_gui:
            search_color = (0,215,255) if E.shape[0] >= args.search_warn_n else None
            stages = [
                {"name": "Capture",  "ms": max(0.0, (time.perf_counter()-t0)*1000 - (td+t_lmk+t_align+t_emb+t_srch)), "ok": True},
                {"name": "Detect",   "ms": td,      "ok": True},
                {"name": "Landmarks","ms": t_lmk,   "ok": align_enabled and (lmk_model is not None)},
                {"name": "Align",    "ms": t_align, "ok": True},
                {"name": "Embed",    "ms": t_emb,   "ok": True},
                {"name": "Search",   "ms": t_srch,  "ok": True, "color": search_color},
            ]
            next_y = draw_pipeline(panel, stages)
            if best_face_for_panel is not None:
                draw_topk(panel, best_face_for_panel, thumbs, start_y=max(next_y, 120), row_h=78, tau=tau)
            else:
                draw_text(panel, "Nessun volto rilevato", (18, max(next_y, 120)), 0.6, (80,80,80), 2)

        # FPS
        dt = time.perf_counter()-t0
        fps_hist.append(1.0/max(dt,1e-6))
        fps = sum(fps_hist)/len(fps_hist)

        if not args.no_gui:
            draw_text(panel, f"FPS: {fps:.1f}", (panel.shape[1]-120, 24), 0.7, (40,40,40), 2)
            combined = np.hstack([frame, panel])
            imshow_fit(WIN, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                tau = min(1.0, tau + 0.02)
            elif key == ord('-'):
                tau = max(0.0, tau - 0.02)
            elif key == ord('a'):
                align_enabled = not align_enabled
            elif key == ord('p'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                outp = Path(f"screenshot_{ts}.png")
                cv2.imwrite(str(outp), combined)
                print(f"[saved] {outp}")
            elif key == ord('n'):
                try:
                    newn = input("Nuovo enroll name: ").strip()
                    if newn:
                        print(f"[enroll] name -> '{newn}'")
                        args.enroll_name = newn
                except Exception:
                    print("[enroll] rename fallito (input console non disponibile).")
            elif key == ord('s'):
                gallery_dirty = True
                do_save()
            elif key == ord('e'):
                if best_crop is None or best_emb is None:
                    print("[enroll] nessun volto nel frame.")
                else:
                    path_str = f"enroll_{args.enroll_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    if args.enroll_dir:
                        subdir = Path(args.enroll_dir) / args.enroll_name
                        subdir.mkdir(parents=True, exist_ok=True)
                        img_path = subdir / path_str
                        cv2.imwrite(str(img_path), best_crop)
                        path_to_store = str(img_path)
                    else:
                        path_to_store = path_str

                    E = np.vstack([E, best_emb.reshape(1, -1).astype(np.float32)])
                    labels = np.append(labels, np.array([args.enroll_name], dtype=object))
                    paths = np.append(paths, np.array([path_to_store], dtype=object))
                    print(f"[enroll] aggiunto '{args.enroll_name}' ({path_to_store}); gallery={E.shape[0]}")
                    gallery_dirty = True
                    do_save()  # autosave immediato
            elif key == ord('d'):
                # delete: nome specifico o ultimo
                try:
                    target = input("Nome da cancellare (vuoto = ultimo): ").strip()
                    if target:
                        idxs = [i for i, l in enumerate(labels) if l == target]
                        if idxs:
                            E = np.delete(E, idxs, axis=0)
                            labels = np.delete(labels, idxs, axis=0)
                            paths = np.delete(paths, idxs, axis=0)
                            print(f"[delete] rimossi {len(idxs)} esempi di '{target}'")
                            gallery_dirty = True
                            do_save()
                        else:
                            print(f"[delete] nessun '{target}' trovato.")
                    else:
                        if len(labels) > 0:
                            print(f"[delete] rimosso ultimo: {labels[-1]} ({paths[-1]})")
                            E = E[:-1]
                            labels = labels[:-1]
                            paths = paths[:-1]
                            gallery_dirty = True
                            do_save()
                        else:
                            print("[delete] gallery vuota.")
                except Exception as ex:
                    print("[delete][ERROR]", ex)

        else:
            print(f"[frame {frame_id:06d}] fps={fps:.1f} faces={len(boxes)}")
            for line in headless_lines: print("  ", line)
            if save_dir:
                out = frame.copy()
                for (x1,y1,x2,y2,conf) in boxes:
                    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imwrite(str(save_dir / f"frame_{frame_id:06d}.jpg"), out)

        frame_id += 1

    cap.release()
    if not args.no_gui:
        try: cv2.destroyAllWindows()
        except Exception: pass
    if not args.no_autosave:
        do_save()

if __name__ == "__main__":
    main()
