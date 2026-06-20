import pandas as pd
import numpy as np
import os
from river import forest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score

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

# Filtrar solo Hospital para reducir tiempo de ablation
df = df[df["Class"] == "Hospital"].copy().reset_index(drop=True)

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

# Columnas de clase
class_cols = [c for c in X.columns if c.startswith("Class_")]

# =========================
# COMBINACIONES DE FEATURES
# =========================
combinaciones = {
    "1_solo_facility": [col],
    "2_facility_lag1": [col, "lag_1"],
    "3_facility_lag1_lag24": [col, "lag_1", "lag_24"],
    "4_facility_lags": [col, "lag_1", "lag_24", "lag_168"],
    "5_facility_lags_rolling": [col, "lag_1", "lag_24", "lag_168",
                                 "roll_mean_24", "roll_std_24",
                                 "roll_mean_168", "diff_1"],
    "6_todas_con_class": [col, "lag_1", "lag_24", "lag_168",
                           "roll_mean_24", "roll_std_24",
                           "roll_mean_168", "diff_1"] + class_cols
}

# =========================
# ABLATION STUDY
# =========================
print("\n===== ABLATION STUDY - ARF Prequential =====\n")
print(f"{'Combinación':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 82)

resultados = []

for nombre, features in combinaciones.items():

    feats = [f for f in features if f in X.columns]

    X_feats = X[feats]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feats)

    model_ab = forest.ARFClassifier(n_models=5, seed=42)

    y_true_all = []
    y_pred_all = []
    y_scores_all = []

    for i in range(len(X_scaled)):
        x = dict(zip(feats, X_scaled[i]))
        y_true = int(y.iloc[i])

        if i > 0:
            y_pred  = model_ab.predict_one(x) or 0
            y_score = model_ab.predict_proba_one(x).get(1, 0.0)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            y_scores_all.append(y_score)

        model_ab.learn_one(x, y_true)

    acc  = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec  = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1   = f1_score(y_true_all, y_pred_all, zero_division=0)
    auc  = roc_auc_score(y_true_all, y_scores_all)

    print(f"{nombre:<30} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    resultados.append({
        "combinacion": nombre,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    })

# =========================
# GUARDAR RESULTADOS
# =========================
os.makedirs("results", exist_ok=True)

pd.DataFrame(resultados).to_csv(
    "results/ablation_arf.csv",
    index=False
)

print("\nResultados guardados en results/ablation_arf.csv")