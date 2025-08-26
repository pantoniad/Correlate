import pandas as pd

def df_to_latex_table(df: pd.DataFrame, filename: str, add_header=True):
    with open(filename, "w", encoding="utf-8") as f:
        if add_header:
            header = " & ".join(df.columns) + r" \\ \hline"
            f.write(header + "\n")
        for _, row in df.iterrows():
            line = " & ".join(map(str, row.values)) + r" \\"
            f.write(line + "\n")

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Score": [95, 87, 78]
}
df = pd.DataFrame(data)

df_to_latex_table(df, "table_data.tex")
