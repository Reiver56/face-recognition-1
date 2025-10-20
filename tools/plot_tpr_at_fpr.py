import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from pathlib import Path

data = np.load("figs/sims_data.npz")
sims = data["sims"]
labels = data["labels"]

fpr, tpr, thr = metrics.roc_curve(labels, sims)
plt.figure(figsize=(7,5))
plt.plot(thr, tpr, label="TPR", color='green')
plt.plot(thr, 1 - fpr, label="1 - FPR", color='red')
plt.xlabel("Soglia (τ)")
plt.ylabel("Tasso")
plt.title("TPR e (1-FPR) in funzione della soglia")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("figs/tpr_at_fpr.png", dpi=200)
print("[done] Salvato figs/tpr_at_fpr.png")
