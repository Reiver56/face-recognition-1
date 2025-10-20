import matplotlib.pyplot as plt

stages = ["Detect", "Landmarks", "Align", "Embed", "Search"]
times_ms = [3.2, 0.5, 0.2, 120.4, 1.1] 

plt.figure(figsize=(7,4))
plt.bar(stages, times_ms, color='cornflowerblue')
plt.ylabel("Tempo medio (ms)")
plt.title("Latenza media per stage")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figs/pipeline_latency.png", dpi=200)
print("[done] Salvato figs/pipeline_latency.png")
