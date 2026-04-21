import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
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
df["theft"] = (df["theft"] != "Normal").astype(int)         # Transformación de variable objetivo, Normal = 0, Resto = 1 (Fraude).
df = df.dropna(subset=["theft"])                            # Eliminar valores nulos en la variable objetivo.

df = pd.get_dummies(df, columns=["Class"])                  # Convertir categorías en variables numéricas, One-Hot encoding.

# Separar variables
X = df.drop("theft", axis=1).values                         # X → variables predictoras.
y = df["theft"].values                                      # y → variable objetivo.

# Escalado
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Parámetros del modelo y del flujo
n_neighbors = 5
batch_size = 1000
window_size = 50000                                         # tamaño máximo de la memoria

# Almacenamiento de resultados acumulados
y_true_all = []
y_pred_all = []
y_scores_all = []

# Conjunto de referencia inicial
X_ref = None
y_ref = None

# Simulación del flujo online
for i in range(0, len(X), batch_size):
    X_batch = X[i:i + batch_size]
    y_batch = y[i:i + batch_size]

    # Primer bloque: solo inicializa la memoria
    if i == 0:
        X_ref = X_batch.copy()
        y_ref = y_batch.copy()
        continue

    # Crear modelo kNN con la memoria actual
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        metric="minkowski",
        p=2,
        algorithm="auto",
        n_jobs=-1
    )

    # Ajuste del modelo sobre la memoria disponible
    model.fit(X_ref, y_ref)

    # Predicción antes de actualizar la memoria
    y_pred = model.predict(X_batch)
    y_score = model.predict_proba(X_batch)[:, 1]

    # Guardar resultados
    y_true_all.extend(y_batch)
    y_pred_all.extend(y_pred)
    y_scores_all.extend(y_score)

    # Actualización de la memoria con ventana deslizante
    X_ref = np.vstack((X_ref, X_batch))[-window_size:]
    y_ref = np.concatenate((y_ref, y_batch))[-window_size:]

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
plt.plot(fpr_curve, tpr_curve, label=f"kNN Online (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC - kNN Online")
plt.legend()
plt.grid()

plt.savefig("results/roc_knn_online.png", dpi=300)
plt.close()

print("\n===== Resultados kNN Online =====")
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
    "model": "knn_online",
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