import pandas as pd
import os
import matplotlib.pyplot as plt

from river import forest

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report,
    roc_auc_score,
    roc_curve
)

os.makedirs("results", exist_ok=True)

# Cargar dataset
df = pd.read_csv("data/df.csv", index_col=0)

# Preprocesamiento
df["theft"] = df["theft"].str.strip()
df["theft"] = (df["theft"] != "Normal").astype(int)
df = df.dropna(subset=["theft"])

df = pd.get_dummies(df, columns=["Class"])

# Separar variables
X = df.drop("theft", axis=1)
y = df["theft"]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir a formato river
X_stream = [dict(zip(X.columns, row)) for row in X_scaled]

# Crear modelo Adaptive Random Forest
model = forest.ARFClassifier(
    n_models=10,
    seed=42
)

# Almacenamiento de resultados
y_true_all = []
y_pred_all = []
y_scores_all = []

# Flujo online
for i in range(len(X_stream)):
    x = X_stream[i]
    y_true = y.iloc[i]

    # Predicción antes de entrenar
    if i > 0:
        y_pred = model.predict_one(x)
        y_score = model.predict_proba_one(x).get(1, 0)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred if y_pred is not None else 0)
        y_scores_all.append(y_score)

    # Entrenamiento incremental
    model.learn_one(x, y_true)

# Métricas finales
accuracy = accuracy_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all, zero_division=0)
recall = recall_score(y_true_all, y_pred_all, zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
cm = confusion_matrix(y_true_all, y_pred_all)
kappa = cohen_kappa_score(y_true_all, y_pred_all)

TN, FP, FN, TP = cm.ravel()
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
roc_auc = roc_auc_score(y_true_all, y_scores_all)

# Curva ROC
fpr_curve, tpr_curve, _ = roc_curve(y_true_all, y_scores_all)

plt.figure()
plt.plot(fpr_curve, tpr_curve, label=f"Adaptive Random Forest (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC - Adaptive Random Forest")
plt.legend()
plt.grid()

plt.savefig("results/roc_arf.png", dpi=300)
plt.close()

print("\n===== Resultados Adaptive Random Forest =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Cohen Kappa:", kappa)
print("FPR:", fpr)
print("ROC-AUC:", roc_auc)
print("\nMatriz de Confusión:")
print(cm)

print(classification_report(y_true_all, y_pred_all, zero_division=0))

# Guardar resultados
results = {
    "model": "adaptive_random_forest",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "kappa": kappa,
    "fpr": fpr,
    "roc_auc": roc_auc,
    "cv_mean_f1": None,
    "cv_std_f1": None
}

results_df = pd.DataFrame([results])

results_file = "results/model_results.csv"

if os.path.exists(results_file):
    results_df.to_csv(results_file, mode="a", header=False, index=False)
else:
    results_df.to_csv(results_file, index=False)