# src/arcface_ov.py
from pathlib import Path
import numpy as np
import cv2
from openvino.runtime import Core

def _dims(ps):
    out = []
    for d in ps:
        out.append(int(d.get_length()) if d.is_static else -1)
    return out

def _detect_layout(inp):
    dims = _dims(inp.partial_shape)
    layout = "NCHW"
    if len(dims) == 4:
        c, last = dims[1], dims[-1]
        if (c not in (1,3)) and (last in (1,3)):
            layout = "NHWC"
    return layout

def _pick_embedding_output(model):
    cand2d, cand4d = [], []
    for o in model.outputs:
        sh = _dims(o.partial_shape)
        if len(sh) == 2 and sh[-1] in (512, 256):
            cand2d.append((o, sh))
        elif len(sh) == 4 and (512 in sh or 256 in sh):
            cand4d.append((o, sh))
    if cand2d: return cand2d[0][0], "2d"
    if cand4d: return cand4d[0][0], "4d"
    return model.outputs[0], "unknown"

class ArcFaceOV:
    """
    Backbone ArcFace (OpenVINO).
    Config raccomandata per OMZ resnet100-arcface-onnx:
      - BGR (swaprb=False)
      - preproc='raw255' (niente normalizzazione: il modello la fa dentro)
      - input 112x112
    """
    def __init__(self, xml_path, swaprb=False, preproc="raw255"):
        self.swaprb = bool(swaprb)
        self.preproc = preproc
        self.core = Core()
        model = self.core.read_model(str(xml_path))
        self.out_port, self.out_kind = _pick_embedding_output(model)
        self.compiled = self.core.compile_model(
            model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": "AUTO"}
        )
        self.inp_port = self.compiled.input(0)
        self.out_port = self.compiled.output(self.out_port.get_index())
        self.layout = _detect_layout(model.inputs[0])
        self.req = self.compiled.create_infer_request()

    def _preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (112, 112))
        if self.preproc == "raw255":
            blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(112,112),
                                         mean=(0,0,0), swapRB=self.swaprb, crop=False)
        elif self.preproc == "arcface":
            # (sconsigliato per questo OMZ: tende a collassare le feature)
            blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/127.5, size=(112,112),
                                         mean=(127.5,127.5,127.5), swapRB=self.swaprb, crop=False)
        else:
            raise ValueError(f"preproc sconosciuto: {self.preproc}")
        x = blob if self.layout == "NCHW" else np.transpose(blob, (0,2,3,1))
        return x.astype(np.float32)

    def embed_bgr(self, img_bgr):
        x = self._preprocess(img_bgr)
        res = self.req.infer({self.inp_port: x})
        out = res[self.out_port]
        if out.ndim == 4:
            out = out.mean(axis=(2,3))  # GAP
        elif out.ndim == 3:
            out = out.mean(axis=2)
        feat = out.reshape(-1).astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-12)
        return feat

    def embed_path(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Impossibile leggere immagine: {path}")
        return self.embed_bgr(img)
