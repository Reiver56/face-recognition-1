import cv2, numpy as np
from pathlib import Path

# importo modello e immagine croppata
MODEL = "models/arcfaceresnet100-8.onnx"
IMG = "out/crop_0.png"


def arcface_embed(net, bgr):
    # Preprocess (ArcFace standard): 112x112, RGB, mean=128.5, std=127.5
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb, (112,112))
    blob = cv2.dnn.blobFromImage(face, 1/127.5, (112,112), (127.5,127.5,127.5), swapRB=False, crop=False)

    # Inference
    net.setInput(blob)
    feat = net.forward().reshape(-1).astype(np.float32) # (512,)
    feat /= (np.linalg.norm(feat) + 1e-12)  # L2-normalize
    return feat

def main():
    if not Path(MODEL).exists():
        print(f"Model not found: {MODEL}")
        return
    if not Path(IMG).exists():
        print(f"Image not found: {IMG}")
        return
    
    net = cv2.dnn.readNet(MODEL)
    img = cv2.imread(IMG)
    emb = arcface_embed(net, img)
    print("Embedding shape:", emb.shape)
    print("L2 norm:", np.linalg.norm(emb))
    np.save("out/emb.npy", emb)
    print("Embedding saved to out/emb.npy")

if __name__ == "__main__":
    import cv2
    main()