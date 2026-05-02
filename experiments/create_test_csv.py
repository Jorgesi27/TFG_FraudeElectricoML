import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Cargar dataset original
df = pd.read_csv("data/df.csv", index_col=0)

# Crear CSV pequeño para probar la API
df_test = df.head(1000).copy()

# En un caso real el operador no tendría la etiqueta theft,
# así que la quitamos para simular datos nuevos sin etiquetar.
if "theft" in df_test.columns:
    df_test = df_test.drop(columns=["theft"])

# Guardar CSV de prueba
df_test.to_csv("data/test_prediccion_1000.csv", index=False)

print("CSV de prueba creado en: data/test_prediccion_1000.csv")
print("Filas:", len(df_test))
print("Columnas:", list(df_test.columns))