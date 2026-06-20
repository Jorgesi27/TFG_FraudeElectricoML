import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
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
# COLUMNAS DE SUBSISTEMAS (candidatas a re-incorporar)
# =========================
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
cols_subsistemas = [c for c in cols_subsistemas if c in df.columns]

# =========================
# FEATURE TEMPORAL — PASADO (lags, ya existentes)
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
# FEATURE TEMPORAL — FUTURO (leads, nuevas)
# Solo válido en offline: el modelo ve el año completo de antemano
# =========================
df["lead_1"] = df.groupby(["Class", "year_block"])[col].shift(-1)
df["lead_24"] = df.groupby(["Class", "year_block"])[col].shift(-24)

df["roll_mean_24_forward"] = df.groupby(["Class", "year_block"])[col].transform(
    lambda x: x.shift(-23).rolling(24).mean()
)

df["diff_lead_1"] = df["lead_1"] - df[col]

# =========================
# LIMPIEZA
# =========================
df = df.dropna().reset_index(drop=True)

# =========================
# SPLIT TEMPORAL
# =========================
train_df = df[df["year_block"] < 3].copy()
test_df  = df[df["year_block"] == 3].copy()

# =========================
# ONE HOT ENCODING
# =========================
train_df = pd.get_dummies(train_df, columns=["Class"], drop_first=True)
test_df  = pd.get_dummies(test_df,  columns=["Class"], drop_first=True)

# =========================
# SEPARAR TARGET Y FEATURES
# =========================
drop_cols = ["theft", "target", "row_in_class", "year_block"]

X_train = train_df.drop(columns=drop_cols)
X_test  = test_df.drop(columns=drop_cols)

y_train = train_df["target"]
y_test  = test_df["target"]

X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

# =========================
# BALANCEO DE CLASES
# =========================
scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

class_cols = [c for c in X_train.columns if c.startswith("Class_")]

# Features base ya validadas en el ablation anterior
base_pasado = [col, "lag_1", "lag_24", "lag_168",
               "roll_mean_24", "roll_std_24",
               "roll_mean_168", "diff_1"] + class_cols

# =========================
# COMBINACIONES DE FEATURES
# =========================
combinaciones = {
    "1_base_pasado": base_pasado,

    "2_base_mas_lead1": base_pasado + ["lead_1"],

    "3_base_mas_leads": base_pasado + ["lead_1", "lead_24",
                                        "roll_mean_24_forward", "diff_lead_1"],

    "4_base_mas_subsistemas": base_pasado + cols_subsistemas,

    "5_base_leads_subsistemas": base_pasado + ["lead_1", "lead_24",
                                                 "roll_mean_24_forward", "diff_lead_1"]
                                              + cols_subsistemas,
}

# =========================
# ABLATION STUDY
# =========================
print("\n===== ABLATION STUDY EXTENDIDO - XGBoost (leads + subsistemas) =====\n")
print(f"{'Combinación':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 84)

resultados = []

for nombre, features in combinaciones.items():

    feats = [f for f in features if f in X_train.columns]

    model_ab = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model_ab.fit(X_train[feats], y_train)
    y_pred_ab = model_ab.predict(X_test[feats])
    y_prob_ab = model_ab.predict_proba(X_test[feats])[:, 1]

    acc  = accuracy_score(y_test, y_pred_ab)
    prec = precision_score(y_test, y_pred_ab, zero_division=0)
    rec  = recall_score(y_test, y_pred_ab, zero_division=0)
    f1   = f1_score(y_test, y_pred_ab, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob_ab)

    print(f"{nombre:<32} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")

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
    "results/ablation_xgboost_extended.csv",
    index=False
)

print("\nResultados guardados en results/ablation_xgboost_extended.csv")