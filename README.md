# Face Recognition (ArcFace + OpenVINO)

Pipeline completa per face recognition con:
- **ArcFace** (embedding 512-D) convertito in **OpenVINO**
- **Face detection** (intel/face-detection-retail-0004) + **landmarks** (intel/landmarks-regression-retail-0009)
- Valutazione con **ROC/AUC/EER**
- **Webcam demo** con overlay pipeline, top-K, **enrollment persistente** e salvataggio dell’indice
- Supporto **Windows**, **Linux/WSL**, CPU
---



---

## Requisiti

- Python 3.10–3.12 consigliato
- CPU x86_64
- OpenVINO Runtime
- OpenCV

Installazione (Linux/macOS):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```
Installazione (Windows PowerShell):
```
python -m venv myvenv
myvenv\Scripts\Activate.ps1
pip install --upgrade pip wheel
pip install -r requirements.txt
```
Modelli

Scarica (Open Model Zoo o bundle tuo) e imposta i path in configs/arcface_openvino.json, es.:
```
{
  "xml": "public/face-recognition-resnet100-arcface-onnx/FP32/face-recognition-resnet100-arcface-onnx.xml",
  "preproc": "raw255",
  "swaprb": false,
  "thresholds": { "default": 0.30 }
}

```
Detector: intel/face-detection-retail-0004
Landmarks: intel/landmarks-regression-retail-0009

Allineamento dataset

Se parti da immagini grezze:
```
python tools/align_dataset_omz.py \
  --det-xml intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml \
  --lmk-xml intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
  --src data/raw/lfw \
  --dst data/aligned/lfw

```
Output: cartelle data/aligned/<dataset>/<identity>/*.jpg 112×112.

Costruzione gallery (.npz)
```
python tools/build_gallery.py \
  --config configs/arcface_openvino.json \
  --root data/aligned/lfw \
  --out data/index/lfw_arcface_raw255_bgr.npz
```


Output:

embeddings (N×512 float32, normalizzati)

labels (N oggetti)

paths (N path)

Valutazione (AUC/EER)
```
python src/eval_verify_openvino.py \
  --xml public/face-recognition-resnet100-arcface-onnx/FP32/face-recognition-resnet100-arcface-onnx.xml \
  --pairs 60000 --swaprb 0 --preproc raw255 --split lfw
```

Stampa AUC, EER, soglia a EER, TPR@FPR target.

Demo Webcam + Enrollment
Windows (PowerShell)
```
python demo\webcam_identify_ui.py `
  --config configs\arcface_openvino.json `
  --index data\index\lfw_arcface_raw255_bgr.npz `
  --det-xml intel\face-detection-retail-0004\FP32\face-detection-retail-0004.xml `
  --lmk-xml intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml `
  --tau 0.30 --topk 5 --source 0 `
  --enroll-dir data\enroll --enroll-name Matteo `
  --index-save data\index\lfw_plus_me.npz
```
Linux
```
python demo/webcam_identify_ui.py \
  --config configs/arcface_openvino.json \
  --index data/index/lfw_arcface_raw255_bgr.npz \
  --det-xml intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml \
  --lmk-xml intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
  --tau 0.30 --topk 5 --source 0 \
  --enroll-dir data/enroll --enroll-name Matteo \
  --index-save data/index/lfw_plus_me.npz
```

Tasti utili

e → Enroll volto migliore del frame (salva crop e embedding, marca index “dirty”)

s → Save manuale dell’indice (--index-save o sovrascrive l’indice caricato)

q → Quit (se autosave abilitato, salva prima di uscire)

'+' / '-' → cambia soglia τ

a → abilita/disabilita align 5-point

p → screenshot finestra
