import os
import pickle
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

models = {
    "Logistic Regression": "results/roc_logistic_regression.pkl",
    "Random Forest": "results/roc_random_forest.pkl",
    "Gradient Boosting": "results/roc_gradient_boosting.pkl",
    "kNN": "results/roc_knn.pkl",
    "XGBoost": "results/roc_xgboost.pkl",
    "MLP": "results/roc_mlp.pkl"
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
linestyles = ["-", "--", "-.", ":", "-", "--"]

plt.figure(figsize=(10, 7))

for (label, path), color, style in zip(models.items(), colors, linestyles):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        linewidth = 3 if "XGBoost" in label else 2

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
plt.title("Comparación de curvas ROC - Modelos Offline", fontsize=14)

plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("results/roc_comparison_offline.png", dpi=300)
plt.show()
plt.close()