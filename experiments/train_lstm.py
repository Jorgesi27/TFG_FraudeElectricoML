import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score,
    classification_report, roc_auc_score, roc_curve
)

os.makedirs("results", exist_ok=True)

# =========================
# CARGA Y PREPROCESAMIENTO
# =========================
df = pd.read_csv("data/df.csv")
df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
df = df[df["Class"] != "0"].copy()

if "0" in df.columns:
    df = df.drop(columns=["0"])

df["target"] = (df["theft"] != "Normal").astype(int)

df["row_in_class"] = df.groupby("Class").cumcount()
df["year_block"] = df["row_in_class"] // 8760

col = "Electricity_Facility__kW__Hourly_"

df["lag_1"]   = df.groupby(["Class", "year_block"])[col].shift(1)
df["lag_24"]  = df.groupby(["Class", "year_block"])[col].shift(24)
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

year_block = df["year_block"].values

X = df.drop(columns=drop_cols).values.astype(np.float32)
y = df["target"].values.astype(np.float32)

# =========================
# ESCALADO
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT TEMPORAL
# =========================
train_mask = year_block < 3
test_mask  = year_block == 3

X_train_flat = X_scaled[train_mask]
X_test_flat  = X_scaled[test_mask]
y_train      = y[train_mask]
y_test       = y[test_mask]

# =========================
# CONSTRUCCIÓN DE SECUENCIAS LSTM
# Ventana de 72h (3 días) — equilibrio entre contexto temporal
# =========================
WINDOW = 72

def build_sequences(X_flat, y_arr, window):
    Xs, ys = [], []
    for i in range(window, len(X_flat)):
        Xs.append(X_flat[i - window:i])
        ys.append(y_arr[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

print("Construyendo secuencias de entrenamiento...")
X_train_seq, y_train_seq = build_sequences(X_train_flat, y_train, WINDOW)

print("Construyendo secuencias de test...")
X_test_seq, y_test_seq = build_sequences(X_test_flat, y_test, WINDOW)

print(f"Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

# =========================
# BALANCEO DE CLASES
# =========================
n_neg = np.sum(y_train_seq == 0)
n_pos = np.sum(y_train_seq == 1)
class_weight = {0: 1.0, 1: n_neg / n_pos}

# =========================
# MODELO LSTM
# 64 unidades para capturar dependencias en ventana de 72 pasos
# =========================
n_features = X_train_seq.shape[2]

model = Sequential([
    LSTM(
        64,
        input_shape=(WINDOW, n_features),
        return_sequences=False
    ),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# ENTRENAMIENTO
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nEntrenando LSTM...")

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=20,
    batch_size=4096,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# PREDICCIÓN
# =========================
print("\nGenerando predicciones...")

y_prob = model.predict(X_test_seq, batch_size=4096, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# =========================
# MÉTRICAS
# =========================
accuracy  = accuracy_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred, zero_division=0)
recall    = recall_score(y_test_seq, y_pred, zero_division=0)
f1        = f1_score(y_test_seq, y_pred, zero_division=0)
cm        = confusion_matrix(y_test_seq, y_pred)
TN, FP, FN, TP = cm.ravel()
kappa     = cohen_kappa_score(y_test_seq, y_pred)
fpr_val   = FP / (FP + TN) if (FP + TN) > 0 else 0
roc_auc   = roc_auc_score(y_test_seq, y_prob)

fpr_curve, tpr_curve, _ = roc_curve(y_test_seq, y_prob)

# =========================
# ROC
# =========================
with open("results/roc_lstm.pkl", "wb") as f:
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
plt.title("ROC - LSTM")
plt.legend()
plt.grid()
plt.savefig("results/roc_lstm.png", dpi=300)
plt.close()

# =========================
# OUTPUT
# =========================
print("\n===== RESULTADOS LSTM =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("Kappa    :", kappa)
print("FPR      :", fpr_val)
print("ROC-AUC  :", roc_auc)
print("\nMatriz de confusión:\n", cm)
print("\nReporte:\n", classification_report(
    y_test_seq, y_pred, zero_division=0
))

# =========================
# GUARDAR RESULTADOS
# =========================
results = {
    "model": "lstm",
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