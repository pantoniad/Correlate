# Generate a human-readable plain-text report that summarizes the dataset and the polynomial
import os, json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import textwrap

# Reuse df and coeffs from earlier cells or create defaults if missing
if 'df' not in globals():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((10, 2)), columns=["Key 1", "Key 2"])

if 'coeffs' not in globals():
    coeffs = {"a": 0.75, "b": -0.20, "c": 1.10, "d": 0.03, "e": -0.50}

def format_unicode_equation(coefs):
    def term(sign, mag, s):
        return f" {sign} {mag:.6g}·{s}"
    a,b,c,d,e = coefs["a"], coefs["b"], coefs["c"], coefs["d"], coefs["e"]
    eq = f"f(x₁, x₂) = {a:.6g}·x₁³"
    eq += term('+' if b>=0 else '-', abs(b), "x₁²·x₂")
    eq += term('+' if c>=0 else '-', abs(c), "x₁·x₂")
    eq += term('+' if d>=0 else '-', abs(d), "x₁·x₂²")
    eq += term('+' if e>=0 else '-', abs(e), "x₂³")
    return eq

def format_latex_equation(coefs):
    a,b,c,d,e = coefs["a"], coefs["b"], coefs["c"], coefs["d"], coefs["e"]
    def s(v, s):
        return (f"+ {abs(v):.6g}\\,{s}" if v>=0 else f"- {abs(v):.6g}\\,{s}")
    return (
        r"f(x_1,x_2) = "
        + f"{a:.6g}\\,x_1^3 "
        + s(b, "x_1^2 x_2 ")
        + s(c, "x_1 x_2 ")
        + s(d, "x_1 x_2^2 ")
        + s(e, "x_2^3")
    )

def generate_text_report(
    data: pd.DataFrame,
    coefs: dict,
    base_dir: str,
    suggested_name: str = "results"
):
    """
    Create a plain-text report summarizing a small dataset and a polynomial function.
    """
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = os.path.join(base_dir, f"report_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"report_{suggested_name}.txt")

    # Dataset summary
    shape = f"{data.shape[0]} rows x {data.shape[1]} columns"
    preview = data.head(5).to_string(index=False)
    desc = data.describe(include='all').to_string()

    # Coefficients table
    coef_df = pd.DataFrame([coefs], columns=["a","b","c","d","e"])
    coef_table = coef_df.to_string(index=False)

    # Displays
    unicode_eq = format_unicode_equation(coefs)
    latex_eq = format_latex_equation(coefs)

    # Build the report text
    lines = []
    lines.append("RESULTS REPORT")
    lines.append("="*70)
    lines.append(f"Generated (UTC): {ts}")
    lines.append("")
    lines.append("1) DATASET SUMMARY")
    lines.append("-"*70)
    lines.append(f"Shape: {shape}")
    lines.append("Columns: " + ", ".join(map(str, data.columns)))
    lines.append("")
    lines.append("Preview (first 5 rows):")
    lines.append(preview)
    lines.append("")
    lines.append("Descriptive statistics:")
    lines.append(desc)
    lines.append("")
    lines.append("2) POLYNOMIAL MODEL")
    lines.append("-"*70)
    lines.append("Form: f(x1, x2) = a*x1^3 + b*x1^2*x2 + c*x1*x2 + d*x1*x2^2 + e*x2^3")
    lines.append("")
    lines.append("Coefficients (a..e):")
    lines.append(coef_table)
    lines.append("")
    lines.append("Display (Unicode):")
    lines.append(unicode_eq)
    lines.append("")
    lines.append("Display (LaTeX):")
    lines.append(latex_eq)
    lines.append("")
    lines.append("Notes:")
    notes = textwrap.fill(
        "This plain-text report is designed for quick human inspection. "
        "Values are shown with default pandas formatting. The Unicode display uses superscripts "
        "and middle dots for readability in plain text. The LaTeX display can be copied into "
        "your thesis to render a typeset equation.", width=80
    )
    lines.append(notes)
    lines.append("")
    lines.append("="*70)
    lines.append("End of report.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_dir, report_path

# Execute and return the path
report_dir, report_path = generate_text_report(df, coeffs, "pytorch tests", "readable_summary")

report_dir, report_path
