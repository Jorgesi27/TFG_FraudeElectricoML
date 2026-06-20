import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
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
# COLUMNAS DE SUBSISTEMAS (se mantienen como features)
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
# FEATURE TEMPORAL — PASADO
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
# FEATURE TEMPORAL — FUTURO
# Solo posible en offline: el modelo dispone del año completo
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
# ONE HOT ENCODING (sobre df completo para k-fold)
# =========================
df_encoded = pd.get_dummies(df, columns=["Class"], drop_first=True)

# =========================
# SEPARAR FEATURES Y TARGET
# =========================
drop_cols = ["theft", "target", "row_in_class", "year_block"]

year_block = df_encoded["year_block"].values

X_all = df_encoded.drop(columns=drop_cols)
y_all = df_encoded["target"]

# =========================
# K-FOLD TEMPORAL (TimeSeriesSplit)
# =========================
tscv = TimeSeriesSplit(n_splits=3)

fold_results = []

print("\n===== K-FOLD TEMPORAL (3 folds) =====\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):

    X_tr = X_all.iloc[train_idx]
    X_te = X_all.iloc[test_idx]
    y_tr = y_all.iloc[train_idx]
    y_te = y_all.iloc[test_idx]

    X_tr, X_te = X_tr.align(X_te, join="left", axis=1, fill_value=0)

    m = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    m.fit(X_tr, y_tr)

    y_pr = m.predict(X_te)
    y_pb = m.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pr)
    prec = precision_score(y_te, y_pr, zero_division=0)
    rec  = recall_score(y_te, y_pr, zero_division=0)
    f1   = f1_score(y_te, y_pr, zero_division=0)
    kap  = cohen_kappa_score(y_te, y_pr)
    auc  = roc_auc_score(y_te, y_pb)
    cm_f = confusion_matrix(y_te, y_pr)
    TN_f, FP_f, FN_f, TP_f = cm_f.ravel()
    fpr_f = FP_f / (FP_f + TN_f) if (FP_f + TN_f) > 0 else 0

    fold_results.append({
        "fold":      fold + 1,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "kappa":     kap,
        "fpr":       fpr_f,
        "auc":       auc
    })

    print(f"Fold {fold+1}: Acc={acc:.4f} Prec={prec:.4f} "
          f"Rec={rec:.4f} F1={f1:.4f} "
          f"Kappa={kap:.4f} FPR={fpr_f:.4f} AUC={auc:.4f}")

fold_df = pd.DataFrame(fold_results)
print("\n===== MEDIA ± STD =====")
for metric in ["accuracy", "precision", "recall", "f1", "kappa", "fpr", "auc"]:
    print(f"{metric:>10}: {fold_df[metric].mean():.4f} ± {fold_df[metric].std():.4f}")

fold_df.to_csv("results/kfold_random_forest.csv", index=False)
print("\nResultados k-fold guardados en results/kfold_random_forest.csv")

# =========================
# MODELO FINAL
# Train: años 1-2-3 / Test: año 4
# =========================
print("\n===== MODELO FINAL (Train años 1-2-3 / Test año 4) =====")

train_df = df_encoded[year_block < 3].copy()
test_df  = df_encoded[year_block == 3].copy()

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

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# PREDICCIÓN
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# MÉTRICAS
# =========================
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
cm        = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
kappa     = cohen_kappa_score(y_test, y_pred)
fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0
roc_auc   = roc_auc_score(y_test, y_prob)
fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)

# =========================
# RESULTADOS / ROC
# =========================
os.makedirs("results", exist_ok=True)

with open("results/roc_random_forest.pkl", "wb") as f:
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
plt.title("ROC - Random Forest")
plt.legend()
plt.grid()
plt.savefig("results/roc_random_forest.png", dpi=300)
plt.close()

# =========================
# OUTPUT
# =========================
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Kappa:", kappa)
print("FPR:", fpr)
print("ROC-AUC:", roc_auc)
print("\nMatriz de confusión:\n", cm)
print("\nReporte:\n", classification_report(y_test, y_pred, zero_division=0))

# =========================
# FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTOP 10 FEATURES:")
print(importance.head(10))

importance.to_csv("results/random_forest_feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
top = importance.head(10)
plt.barh(top["feature"], top["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("results/random_forest_feature_importance.png")
plt.close()

# =========================
# GUARDAR RESULTADOS
# =========================
results = {
    "model": "random_forest",
    "accuracy":  accuracy,
    "precision": precision,
    "recall":    recall,
    "f1_score":  f1,
    "kappa":     kappa,
    "fpr":       fpr,
    "roc_auc":   roc_auc
}

results_df = pd.DataFrame([results])
results_file = "results/model_results.csv"

if os.path.exists(results_file):
    results_df.to_csv(results_file, mode="a", header=False, index=False)
else:
    results_df.to_csv(results_file, index=False)

print("\nResultados guardados en results/model_results.csv")