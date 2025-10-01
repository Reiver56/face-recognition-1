# tools/verify.py
import argparse
import numpy as np
import sys
sys.path.insert(0,'/root/face-recognition-1/src')
from arcface_ov import ArcFaceOV

def cosine(a, b): 
    return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path al modello OpenVINO (.xml)")
    ap.add_argument("--img1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--tau", type=float, default=0.30, help="Soglia decisione (default 0.30)")
    ap.add_argument("--swaprb", type=int, choices=[0,1], default=0)
    ap.add_argument("--preproc", choices=["raw255","arcface"], default="raw255")
    args = ap.parse_args()

    model = ArcFaceOV(args.xml, swaprb=bool(args.swaprb), preproc=args.preproc)
    f1 = model.embed_path(args.img1)
    f2 = model.embed_path(args.img2)
    s = cosine(f1, f2)
    match = s >= args.tau
    print(f"score (cosine) = {s:.4f} | tau = {args.tau:.3f} -> {'MATCH' if match else 'NO MATCH'}")

if __name__ == "__main__":
    main()