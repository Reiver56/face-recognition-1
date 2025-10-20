#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera pairs_auto.txt per LFW in formato:
  - 3 token: "ID fileA.jpg fileB.jpg"  (positivi)
  - 4 token: "IDa fileA.jpg IDb fileB.jpg" (negativi)
a partire dalla struttura:
root/
  Person_1/*.jpg
  Person_2/*.jpg
  ...
"""

import argparse, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory LFW (cartelle per identità)")
    ap.add_argument("--out", default="data/lfw/pairs_auto.txt", help="Output pairs")
    ap.add_argument("--pos-per-id", type=int, default=2, help="coppie positive per identità (max combinazioni)")
    ap.add_argument("--neg", type=int, default=3000, help="numero di coppie negative totali")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    random.seed(args.seed)

    # raccogli immagini per identità
    by_id = {}
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir(): 
            continue
        imgs = sorted([p for p in person_dir.glob("*.jpg") if p.is_file()])
        if len(imgs) >= 2:
            by_id[person_dir.name] = imgs

    ids = list(by_id.keys())
    print(f"[info] identità con >=2 immagini: {len(ids)}")

    lines = []

    # POSITIVE: per ogni id prendi alcune coppie (senza ripetizioni)
    for pid in ids:
        imgs = by_id[pid]
        # tutte le coppie possibili
        all_pairs = [(imgs[i], imgs[j]) for i in range(len(imgs)) for j in range(i+1, len(imgs))]
        random.shuffle(all_pairs)
        take = min(args.pos_per_id, len(all_pairs))
        for a,b in all_pairs[:take]:
            # formato a 3 token: "ID fileA fileB"
            lines.append(f"{pid} {a.name} {b.name}")

    # NEGATIVE: estrai a caso due identità diverse e 1 immagine ciascuna
    for _ in range(args.neg):
        id_a, id_b = random.sample(ids, 2)
        a = random.choice(by_id[id_a]).name
        b = random.choice(by_id[id_b]).name
        # formato a 4 token: "IDa fileA IDb fileB"
        lines.append(f"{id_a} {a} {id_b} {b}")

    # mescola e salva
    random.shuffle(lines)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] pairs salvato in: {outp}  (tot={len(lines)}, pos~{len(ids)*args.pos_per_id}, neg={args.neg})")

if __name__ == "__main__":
    main()
