import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
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

os.makedirs("results", exist_ok=True)

# Cargar dataset
df = pd.read_csv("data/df.csv", index_col=0)

# Convertir variables categóricas
df["theft"] = df["theft"].str.strip()
df["theft"] = (df["theft"] != "Normal").astype(int)         # Transformación de variable objetivo, Normal = 0, Resto = 1 (Fraude).
df = df.dropna(subset=["theft"])                            # Eliminar valores nulos en la variable objetivo.

df = pd.get_dummies(df, columns=["Class"])                  # Convertir categorías en variables numéricas, One-Hot encoding.

# Separar variables
X = df.drop("theft", axis=1)                                # X → variables predictoras.
y = df["theft"]                                             # y → variable objetivo.

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear modelo (pipeline) con la distancia euclídea
pipeline = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2, algorithm="auto", n_jobs=-1))])

# Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")

# Entrenar modelo con balanceo
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

plt.figure()
plt.plot(fpr_curve, tpr_curve, label=f"kNN (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC - kNN")
plt.legend()
plt.grid()

# Guardar imagen
plt.savefig("results/roc_knn.png", dpi=300)
plt.close()

# Resultados
print("\n===== Resultados kNN =====")
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

# Guardar resultados
results = {
    "model": "kNN",
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