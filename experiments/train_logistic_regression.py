import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# Almacenamiento curvas ROC (Solo se añade en el script final)
roc_data = []

# Cargar dataset
df = pd.read_csv("data/df.csv", index_col=0)

# Convertir variables categóicas
df["theft"] = df["theft"].str.strip()
df["theft"] = (df["theft"] != "Normal").astype(int)
df = df.dropna(subset=["theft"])

df = pd.get_dummies(df, columns=["Class"])

# Separar variables
X = df.drop("theft", axis=1)
y = df["theft"]


# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear modelo
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])

# Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Predicciones
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
roc_auc = roc_auc_score(y_test, y_prob)

# Curva ROC
fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)

# Guardar para comparativa con los otros métodos (Solo se añade en el script final)
roc_data.append(("Logistic Regression", fpr_curve, tpr_curve, roc_auc))

plt.figure()
plt.plot(fpr_curve, tpr_curve, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC - Regresión Logística")
plt.legend()
plt.grid()

# Guardar imagen
os.makedirs("results", exist_ok=True)
plt.savefig("results/roc_logistic.png", dpi=300)
plt.close()

# Resultados
print("\n===== Resultados Regresión Logística =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Cohen Kappa:", kappa)
print("FPR:", fpr)
print("ROC-AUC:", roc_auc)

print("\nMatriz de Confusión:")
print(cm)

print(classification_report(y_test, y_pred, zero_division=0))

print("Cross Validation F1 scores:", scores)
print("Mean F1:", scores.mean())
print("Std:", scores.std())

# Guardado de resultados
results = {
    "model": "logistic_regression",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "kappa": kappa,
    "fpr": fpr,
    "roc_auc": roc_auc,
    "cv_mean_f1": scores.mean(),
    "cv_std_f1": scores.std()
}

results_df = pd.DataFrame([results])

results_file = "results/model_results.csv"

if os.path.exists(results_file):
    results_df.to_csv(results_file, mode="a", header=False, index=False)
else:
    results_df.to_csv(results_file, index=False)

# Curva ROC comparativa
plt.figure()

for name, fpr_c, tpr_c, auc_val in roc_data:
    plt.plot(fpr_c, tpr_c, label=f"{name} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparación curvas ROC")
plt.legend()
plt.grid()

plt.savefig("results/roc_comparativa.png", dpi=300)