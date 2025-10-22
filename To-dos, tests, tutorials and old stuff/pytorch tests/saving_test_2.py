# Create a saver for polynomial coefficients with creative displays, then run it
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Example coefficients for f(x1, x2) = a*x1^3 + b*x1^2*x2 + c*x1*x2 + d*x1*x2^2 + e*x2^3
# Use deterministic values for repeatability
coeffs = {
    "a": 0.75,
    "b": -0.20,
    "c": 1.10,
    "d": 0.03,
    "e": -0.50,
}

def format_unicode_equation(coefs):
    # Unicode superscripts: 0 ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹
    # Build term strings with · as multiply and proper superscripts
    parts = []
    a,b,c,d,e = coefs["a"], coefs["b"], coefs["c"], coefs["d"], coefs["e"]
    def fmt_coef(v):
        # Avoid leading + for the first positive term; handle negatives
        return f"{v:.6g}"
    def term(v, s):
        sign = " + " if v >= 0 else " - "
        mag = abs(v)
        return f"{sign}{mag:.6g}·{s}"
    # Start with 'f(x₁, x₂) ='
    eq = "f(x₁, x₂) = " + fmt_coef(a) + "·x₁³"
    eq += term(b, "x₁²·x₂")
    eq += term(c, "x₁·x₂")
    eq += term(d, "x₁·x₂²")
    eq += term(e, "x₂³")
    return eq

def format_latex_equation(coefs):
    a,b,c,d,e = coefs["a"], coefs["b"], coefs["c"], coefs["d"], coefs["e"]
    # LaTeX with aligned signs
    return (
        r"f(x_1,x_2) = "
        + f"{a:.6g}\\,x_1^3 "
        + (f"+ {abs(b):.6g}\\,x_1^2 x_2 " if b >= 0 else f"- {abs(b):.6g}\\,x_1^2 x_2 ")
        + (f"+ {abs(c):.6g}\\,x_1 x_2 "   if c >= 0 else f"- {abs(c):.6g}\\,x_1 x_2 ")
        + (f"+ {abs(d):.6g}\\,x_1 x_2^2 " if d >= 0 else f"- {abs(d):.6g}\\,x_1 x_2^2 ")
        + (f"+ {abs(e):.6g}\\,x_2^3"     if e >= 0 else f"- {abs(e):.6g}\\,x_2^3")
    )

def save_equation_parameters(
    coefs: dict, 
    base_dir: str, 
    suggested_name: str,
    equation_name: str = "f",
    variables=("x1","x2")
) -> str:
    """
    Saves polynomial coefficients and multiple display forms in four formats:
    CSV, JSON, YAML, and SQLite. Files are named <method>_<suggested_name>.*
    
    Polynomial: f(x1,x2) = a*x1^3 + b*x1^2*x2 + c*x1*x2 + d*x1*x2^2 + e*x2^3
    """
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = os.path.join(base_dir, f"poly_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Tidy long-form table of terms for CSV / SQLite
    terms = [
        {"term": "x1^3",   "coef": float(coefs["a"]), "exp_x1": 3, "exp_x2": 0},
        {"term": "x1^2x2", "coef": float(coefs["b"]), "exp_x1": 2, "exp_x2": 1},
        {"term": "x1x2",   "coef": float(coefs["c"]), "exp_x1": 1, "exp_x2": 1},
        {"term": "x1x2^2", "coef": float(coefs["d"]), "exp_x1": 1, "exp_x2": 2},
        {"term": "x2^3",   "coef": float(coefs["e"]), "exp_x1": 0, "exp_x2": 3},
    ]
    df_terms = pd.DataFrame(terms, columns=["term","coef","exp_x1","exp_x2"])

    # Display variants
    unicode_eq = format_unicode_equation(coefs)
    latex_eq = format_latex_equation(coefs)
    
    # 1) CSV
    csv_path = os.path.join(out_dir, f"csv_{suggested_name}.csv")
    df_terms.to_csv(csv_path, index=False, encoding="utf-8")
    
    # 2) JSON
    json_payload = {
        "equation_name": equation_name,
        "variables": list(variables),
        "coefficients": coefs,
        "terms": terms,
        "display": {
            "unicode": unicode_eq,
            "latex": latex_eq
        },
        "schema": {
            "note": "Third-order bivariate polynomial (sparse)",
            "format": "terms with exponents per variable"
        }
    }
    json_path = os.path.join(out_dir, f"json_{suggested_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    
    # 3) YAML (no external dependencies: simple emitter)
    yaml_lines = []
    yaml_lines.append(f"equation_name: {equation_name}")
    yaml_lines.append(f"variables: [{', '.join(variables)}]")
    # coefficients
    yaml_lines.append("coefficients:")
    for k in ["a","b","c","d","e"]:
        yaml_lines.append(f"  {k}: {float(coefs[k])}")
    # terms
    yaml_lines.append("terms:")
    for t in terms:
        yaml_lines.append("  -")
        yaml_lines.append(f"    term: {t['term']}")
        yaml_lines.append(f"    coef: {t['coef']}")
        yaml_lines.append(f"    exp_x1: {t['exp_x1']}")
        yaml_lines.append(f"    exp_x2: {t['exp_x2']}")
    # displays
    yaml_lines.append("display:")
    yaml_lines.append(f"  unicode: |")
    # indent multiline block scalar
    for line in [unicode_eq]:
        yaml_lines.append(f"    {line}")
    yaml_lines.append(f"  latex: |")
    yaml_lines.append(f"    {latex_eq}")
    yaml_path = os.path.join(out_dir, f"yaml_{suggested_name}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")
    
    # 4) SQLite
    sqlite_path = os.path.join(out_dir, f"sqlite_{suggested_name}.sqlite")
    con = sqlite3.connect(sqlite_path)
    try:
        df_terms.to_sql("polynomial_terms", con, if_exists="replace", index=False)
        con.execute("""
            CREATE TABLE IF NOT EXISTS polynomial_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        meta_rows = [
            ("equation_name", equation_name),
            ("variables", json.dumps(list(variables))),
            ("unicode_display", unicode_eq),
            ("latex_display", latex_eq),
            ("form", "a*x1^3 + b*x1^2*x2 + c*x1*x2 + d*x1*x2^2 + e*x2^3")
        ]
        con.executemany("INSERT OR REPLACE INTO polynomial_meta(key, value) VALUES(?,?)", meta_rows)
        con.commit()
    finally:
        con.close()
    
    return out_dir, {"csv": csv_path, "json": json_path, "yaml": yaml_path, "sqlite": sqlite_path}

# Execute saver
out_dir, files = save_equation_parameters(coeffs, "pytorch tests", "coeffs_test")

out_dir, files, format_unicode_equation(coeffs), format_latex_equation(coeffs)
