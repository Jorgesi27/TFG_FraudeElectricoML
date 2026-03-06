import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)


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


# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Crear modelo
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Cross Validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")

# Entrenar modelo
pipeline.fit(X_train, y_train)


# Predicciones
y_pred = pipeline.predict(X_test)


# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# Resultados
print("\n===== Resultados Regresión Logística =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Cohen Kappa:", kappa)

print("\nMatriz de Confusión:")
print(cm)

print(classification_report(y_test, y_pred))

print("Cross Validation F1 scores:", scores)
print("Mean F1:", scores.mean())
print("Std:", scores.std())

results = {
    "model": "logistic_regression",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "kappa": kappa,
    "cv_mean_f1": scores.mean(),
    "cv_std_f1": scores.std()
}

results_df = pd.DataFrame([results])
results_df.to_csv("results/logistic_regression_results.csv", index=False)