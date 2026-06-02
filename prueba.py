import pandas as pd

df = pd.read_csv("data/df.csv", index_col=0)

df = df[df["Class"] != "0"].copy()

df["row_in_class"] = df.groupby("Class").cumcount()
df["year_block"] = df["row_in_class"] // 8760

print(df["year_block"].value_counts().sort_index())