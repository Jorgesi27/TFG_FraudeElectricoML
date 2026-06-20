import pandas as pd
import numpy as np
import os
import pickle
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
    roc_auc_score,
    roc_curve,
    classification_report
)

# =========================
# CARGA
# =========================
df = pd.read_csv("data/df.csv")
df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
df = df[df["Class"] != "0"].copy()

if "0" in df.columns:
    df = df.drop(columns=["0"])

# =========================
# TARGET BINARIO
# =========================
df["target"] = (df["theft"] != "Normal").astype(int)

# =========================
# RECONSTRUCCIÓN TEMPORAL
# =========================
df["row_in_class"] = df.groupby("Class").cumcount()
df["year_block"] = df["row_in_class"] // 8760

# =========================
# FEATURE TEMPORAL
# =========================
col = "Electricity_Facility__kW__Hourly_"

df["lag_1"] = df.groupby(["Class", "year_block"])[col].shift(1)
df["lag_24"] = df.groupby(["Class", "year_block"])[col].shift(24)
df["lag_168"] = df.groupby(["Class", "year_block"])[col].shift(168)

df["roll_mean_24"] = df.groupby(["Class", "year_block"])[col].transform(
    lambda x: x.rolling(24).mean()
)
df["roll_std_24"] = df.groupby(["Class", "year_block"])[col].transform(
    lambda x: x.rolling(24).std()
)
df["roll_mean_168"] = df.groupby(["Class", "year_block"])[col].transform(
    lambda x: x.rolling(168).mean()
)
df["diff_1"] = df[col] - df["lag_1"]

# =========================
# LIMPIEZA
# =========================
df = df.dropna().reset_index(drop=True)

cols_subsistemas = [
    "Fans_Electricity__kW__Hourly_",
    "Cooling_Electricity__kW__Hourly_",
    "Heating_Electricity__kW__Hourly_",
    "InteriorLights_Electricity__kW__Hourly_",
    "InteriorEquipment_Electricity__kW__Hourly_",
    "Gas_Facility__kW__Hourly_",
    "Heating_Gas__kW__Hourly_",
    "InteriorEquipment_Gas__kW__Hourly_",
    "Water_Heater_WaterSystems_Gas__kW__Hourly_"
]
df = df.drop(columns=cols_subsistemas)

# =========================
# MEZCLA TEMPORAL ENTRE EDIFICIOS
# =========================
df = df.sort_values(["year_block", "row_in_class"]).reset_index(drop=True)

# =========================
# ONE HOT ENCODING
# =========================
df = pd.get_dummies(df, columns=["Class"], drop_first=True)

# =========================
# SEPARAR FEATURES Y TARGET
# =========================
drop_cols = ["theft", "target", "row_in_class", "year_block"]

X = df.drop(columns=drop_cols)
y = df["target"]

columns = X.columns.tolist()

# =========================
# ESCALADO
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# MODELO KNN ONLINE — VENTANA DESLIZANTE
# KNN no tiene partial_fit, por lo que se simula el aprendizaje online mediante una ventana deslizante.
# =========================
n_neighbors  = 5
batch_size   = 1000
window_size  = 5000

y_true_all   = []
y_pred_all   = []
y_scores_all = []

X_ref = None
y_ref = None

for i in range(0, len(X_scaled), batch_size):

    X_batch = X_scaled[i:i + batch_size]
    y_batch = y.iloc[i:i + batch_size].values

    # Primer bloque: inicializar memoria
    if i == 0:
        X_ref = X_batch.copy()
        y_ref = y_batch.copy()
        continue

    # Entrenar KNN sobre la ventana actual
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        metric="minkowski",
        p=2,
        algorithm="auto",
        n_jobs=-1
    )
    model.fit(X_ref, y_ref)

    # Predecir antes de actualizar la memoria
    y_pred  = model.predict(X_batch)
    proba   = model.predict_proba(X_batch)
    y_score = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X_batch))

    y_true_all.extend(y_batch)
    y_pred_all.extend(y_pred)
    y_scores_all.extend(y_score)

    # Actualizar ventana deslizante
    X_ref = np.vstack((X_ref, X_batch))[-window_size:]
    y_ref = np.concatenate((y_ref, y_batch))[-window_size:]

    if i % 10000 == 0:
        print(f"Procesados {i}/{len(X_scaled)} registros...")

# =========================
# MÉTRICAS
# =========================
accuracy  = accuracy_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all, zero_division=0)
recall    = recall_score(y_true_all, y_pred_all, zero_division=0)
f1        = f1_score(y_true_all, y_pred_all, zero_division=0)
cm        = confusion_matrix(y_true_all, y_pred_all)
TN, FP, FN, TP = cm.ravel()
kappa     = cohen_kappa_score(y_true_all, y_pred_all)
fpr_val   = FP / (FP + TN) if (FP + TN) > 0 else 0
roc_auc   = roc_auc_score(y_true_all, y_scores_all)
fpr_curve, tpr_curve, _ = roc_curve(y_true_all, y_scores_all)

# =========================
# RESULTADOS / ROC
# =========================
os.makedirs("results", exist_ok=True)

with open("results/roc_knn_online.pkl", "wb") as f:
    pickle.dump({
        "fpr": fpr_curve,
        "tpr": tpr_curve,
        "auc": roc_auc
    }, f)

plt.figure()
plt.plot(fpr_curve, tpr_curve, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - KNN Online")
plt.legend()
plt.grid()
plt.savefig("results/roc_knn_online.png", dpi=300)
plt.close()

# =========================
# OUTPUT
# =========================
print("\n===== RESULTADOS KNN Online =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("Kappa    :", kappa)
print("FPR      :", fpr_val)
print("ROC-AUC  :", roc_auc)
print("\nMatriz de confusión:\n", cm)
print("\nReporte:\n", classification_report(
    y_true_all, y_pred_all, zero_division=0
))

# =========================
# GUARDAR RESULTADOS
# =========================
results = {
    "model": "knn_online",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "kappa": kappa,
    "fpr": fpr_val,
    "roc_auc": roc_auc
}

results_df = pd.DataFrame([results])

results_file = "results/model_results.csv"

if os.path.exists(results_file):
    results_df.to_csv(results_file, mode="a", header=False, index=False)
else:
    results_df.to_csv(results_file, index=False)

print("\nResultados guardados en results/model_results.csv")