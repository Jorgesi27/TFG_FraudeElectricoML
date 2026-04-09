import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
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
df["theft"] = (df["theft"] != "Normal").astype(int)     # Transformación de variable objetivo, Normal = 0, Resto = 1 (Fraude)
df = df.dropna(subset=["theft"])                        # Eliminar valores nulos en la variable objetivo

df = pd.get_dummies(df, columns=["Class"])              # Convertir categorías en variables numéricas, One-Hot encoding.

# Separar variables
X = df.drop("theft", axis=1)                            # X → variables predictoras
y = df["theft"].values                                  # y → variable objetivo

# Escalado
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Crear modelo
model = Perceptron(random_state=42)

# Clases necesarias para el método partial_fit
classes = np.unique(y)

# Simulación del flujo online
batch_size = 1000

# Almacenamiento de resultados acumulados
y_true_all = []
y_pred_all = []
y_scores_all = []

# Entrenamiento incremental
for i in range(0, len(X), batch_size):
    # Simulación de llegada de nuevos datos
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]

    # Predicción antes de entrenar
    if i > 0:
        y_pred = model.predict(X_batch)
        y_score = model.decision_function(X_batch)

        # Resultados para evaluación final
        y_true_all.extend(y_batch)
        y_pred_all.extend(y_pred)
        y_scores_all.extend(y_score)

    # Entrenamiento online
    model.partial_fit(X_batch, y_batch, classes=classes)

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
plt.plot(fpr_curve, tpr_curve, label=f"Perceptrón Online (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC - Perceptrón Online")
plt.legend()
plt.grid()

plt.savefig("results/roc_perceptron_online.png", dpi=300)
plt.close()

print("\n===== Resultados Perceptrón Online=====")
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
    "model": "perceptron_online",
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