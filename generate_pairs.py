# generate_pairs.py
import os
import random
from pathlib import Path

root = Path("data/raw/lfw")
out = Path("data/raw/lfw/pairs.txt")
out.parent.mkdir(parents=True, exist_ok=True)

people = [p for p in root.iterdir() if p.is_dir()]
pairs = []

for person in people:
    imgs = list(person.glob("*.jpg"))
    if len(imgs) >= 2:
        for _ in range(min(3, len(imgs)//2)):
            a, b = random.sample(imgs, 2)
            pairs.append((person.name, a.name, b.name))

# coppie negative
neg_people = random.sample(people, min(len(people), 5))
for i in range(len(neg_people) - 1):
    p1, p2 = neg_people[i], neg_people[i+1]
    i1 = random.choice(list(p1.glob("*.jpg")))
    i2 = random.choice(list(p2.glob("*.jpg")))
    pairs.append((p1.name, i1.name, f"{p2.name}/{i2.name}"))

# scrittura file
with open(out, "w", encoding="utf-8") as f:
    f.write(f"{len(pairs)} 3\n")
    for p in pairs:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

print(f"[ok] pairs.txt generato con {len(pairs)} coppie -> {out}")
