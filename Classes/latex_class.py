import pandas as pd

class latex():

    def __init__(self, df: pd.DataFrame, filename: str, caption: str, label: str, header: pd.DataFrame = [], add_headers = True):

        """
        self

        Inputs:
        -

        Outputs:
        -
        
        """

        self.df = df
        self.filename = filename
        self.caption = caption
        self.label = label
        self.header = header
        self.use_header = add_headers

    def df_to_lxTable(self):
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

        # Extract data from self
        filename = self.filename
        df = self.df
        add_header = self.use_header
        caption = self.caption
        label = self.label

        # Python dataframe to latex table code
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

