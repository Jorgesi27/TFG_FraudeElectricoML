import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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
classes = np.array([0, 1])

# =========================
# ESCALADO
# GaussianNB no requiere escalado estrictamente pero se aplica para mantener coherencia con el resto de modelos online
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# MODELO NAIVE BAYES ONLINE — PREQUENTIAL
# GaussianNB soporta partial_fit para aprendizaje incremental dato a dato
# =========================
model = GaussianNB()

y_true_all = []
y_pred_all = []
y_scores_all = []

warnings.filterwarnings("ignore", category=RuntimeWarning)

for i in range(len(X_scaled)):

    x = X_scaled[i].reshape(1, -1)
    y_true = int(y.iloc[i])

    # Primero predice (si ya ha visto al menos 1 dato)
    if i > 0:
        y_pred  = model.predict(x)[0]
        y_score = model.predict_proba(x)[0][1]

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        y_scores_all.append(y_score)

    # Luego aprende
    model.partial_fit(x, [y_true], classes=classes)

    if i % 10000 == 0:
        print(f"Procesados {i}/{len(X_scaled)} registros...")

# =========================
# LIMPIAR NaN ANTES DE MÉTRICAS
# =========================
y_true_clean = []
y_pred_clean = []
y_scores_clean = []

for yt, yp, ys in zip(y_true_all, y_pred_all, y_scores_all):
    if not np.isnan(ys):
        y_true_clean.append(yt)
        y_pred_clean.append(yp)
        y_scores_clean.append(ys)

y_true_all   = y_true_clean
y_pred_all   = y_pred_clean
y_scores_all = y_scores_clean

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

with open("results/roc_naive_bayes_online.pkl", "wb") as f:
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
plt.title("ROC - Naive Bayes Online")
plt.legend()
plt.grid()
plt.savefig("results/roc_naive_bayes_online.png", dpi=300)
plt.close()

# =========================
# OUTPUT
# =========================
print("\n===== RESULTADOS Naive Bayes Online =====")
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
    "model": "naive_bayes_online",
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