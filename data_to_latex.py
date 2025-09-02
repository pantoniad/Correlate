import pandas as pd

def data_to_latex(df: pd.DataFrame, filename: str, caption: str, label: str, add_header=True):
    """
    data_to_latex: function that exports a dataframe into a styled LaTex table

    Inputs: 
    - df: the dataframe to be exported,
    - filename: the name of the generated file, str,
    - caption: the caption given to the generated table, str,
    - label: the label of the generated table, str, 
    - add_headers: include headers or not, True of False value
    
    Outputs:
    - Generated latex file

    More details: Export a DataFrame into a styled LaTeX table fragment:
    - Wrapped in a table float with caption + label
    - Index column (first col) shaded light gray
    - Bold index and header text
    - Header row shaded light gray
    - Column separators between all columns
    - Double lines at table edges
    - Table width limited to \textwidth (A4 margins) with text wrapping
    """

    with open(filename, "w", encoding="utf-8") as f:
        # Begin table float
        f.write("\\begin{table}[h!]\n")
        f.write("  \\centering\n")

        # Tabularx setup: first column fixed, rest are wrapping (X)
        col_format = "||c|" + "|".join(["X"] * len(df.columns)) + "||"
        f.write(f"  \\begin{{tabularx}}{{\\textwidth}}{{{col_format}}}\n")
        f.write("  \\hline\n")

        # Write header row (with grey background)
        if add_header:
            header_cells = ["\\cellcolor{gray!20}\\textbf{" + (df.index.name or "Parameter") + "}"] \
                           + ["\\cellcolor{gray!20}\\textbf{" + str(c) + "}" for c in df.columns]
            f.write("    " + " & ".join(header_cells) + r" \\ [0.5ex]" + "\n")
            f.write("  \\hline\\hline\n")
        
        # Write body rows
        f.write("\\centering\n")
        for idx, row in df.iterrows():
            row_cells = [f"\\cellcolor{{gray!20}}\\textbf{{{idx}}}"] \
                        + [str(val) for val in row.values]
            f.write("    " + " & ".join(row_cells) + r" \\" + "\n")
            f.write("  \\hline\n")

        f.write("  \\end{tabularx}\n")
        f.write(f"  \\caption{{{caption}}}\n")
        f.write(f"  \\label{{{label}}}\n")
        f.write("\\end{table}\n")

# Engine specs
d = {
    "Thrust rating (kN)": [117],
    "Fan diameter": [1.55],
    "Hub2Tip": [0.3],
    "Bypass ratio": [5.1],
    "Fan PR": [1.6],
    "Booster PR": [1.55],
    "High pressure compressor PR": [11]
}

specs = pd.DataFrame(
    data = d,
    index = ["Parameter", "Value"]
)

data_to_latex(specs, "specs.tex", "CFM56-7B26 specifications", "tab:specs")

# Operating conditions for each point: Altitude, Required thrust, Flight speed, Axial fan speed
d = {
    "Idle": [0, 8.19, 0, 0.09],
    "Take-off": [11, 117, 0.3, 0.4],
    "Climb-out": [305, 99.45, 0.3, 0.4],
    "Approach": [914, 35.1, 0.2, 0.3]
}

lto_ops = pd.DataFrame(
    data = d,
    index = ["Altitude (m)", "Required thrust (kN)", "Flight speed (Mach)", "Axial fan speed (Mach)"]
)

data_to_latex(lto_ops, "lto_ops.tex", "LTO operating conditions", "tab:lto")

