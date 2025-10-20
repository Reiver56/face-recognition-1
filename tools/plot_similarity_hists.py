import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# carica il file salvato da eval_verify_openvino.py
data = np.load("figs/sims_data.npz")
sims = data["sims"]
labels = data["labels"]

plt.figure(figsize=(7,5))
plt.hist(sims[labels == 1], bins=60, alpha=0.6, color='green', label='Positivi')
plt.hist(sims[labels == 0], bins=60, alpha=0.6, color='red', label='Negativi')
plt.xlabel("Cosine similarity")
plt.ylabel("Frequenza")
plt.title("Distribuzione delle similarità")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("figs/hist_similarity.png", dpi=200)
print("[done] Salvato figs/hist_similarity.png")
