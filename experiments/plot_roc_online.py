import os
import pickle
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

models = {
    "Perceptrón Online": "results/roc_perceptron_online.pkl",
    "Naive Bayes Incremental": "results/roc_naive_bayes_incremental.pkl",
    "kNN Online": "results/roc_knn_online.pkl",
    "Hoeffding Tree": "results/roc_hoeffding_tree.pkl",
    "Adaptive Random Forest": "results/roc_adaptive_random_forest.pkl"
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
linestyles = ["-", "--", "-.", ":", "-"]

plt.figure(figsize=(10, 7))

for (label, path), color, style in zip(models.items(), colors, linestyles):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        linewidth = 3 if "Adaptive Random Forest" in label else 2

        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{label} (AUC = {data['auc']:.3f})",
            color=color,
            linestyle=style,
            linewidth=linewidth
        )

# Línea base
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("Comparación de curvas ROC - Modelos Online", fontsize=14)

plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("results/roc_comparison_online.png", dpi=300)
plt.show()
plt.close()