#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Uso: python inspect_index.py <file_index.npz>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERRORE] File non trovato: {path}")
        sys.exit(1)

    try:
        d = np.load(str(path), allow_pickle=True)
        print(f"[OK] Caricato {path}")
        print("Chiavi:", list(d.keys()))
        if "embeddings" in d:
            print("  embeddings.shape:", d["embeddings"].shape)
        if "labels" in d:
            labs = d["labels"]
            print("  labels.len:", len(labs))
            if len(labs) > 10:
                print("  labels[:10]:", labs[:10])
            else:
                print("  labels:", labs)
        if "paths" in d:
            print("  paths.len:", len(d["paths"]))
            print("  esempio path:", d["paths"][0] if len(d["paths"]) else "—")
    except Exception as e:
        print(f"[ERRORE] Caricamento fallito: {e}")
        print("Dimensione file:", os.path.getsize(path), "bytes")

if __name__ == "__main__":
    main()
