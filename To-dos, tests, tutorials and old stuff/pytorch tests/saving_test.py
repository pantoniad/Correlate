# Generate random test data and a save function, then run it
import os, json, sqlite3, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Create the random dataframe (2 columns x 10 rows), reproducible
rng = np.random.default_rng(42)
df = pd.DataFrame(rng.random((10, 2)), columns=["Key 1", "Key 2"])

# Display the dataframe to the user
#from caas_jupyter_tools import display_dataframe_to_user
#display_dataframe_to_user("Random Test Data (Key 1, Key 2)", df)

# Define the saver function
def save_run_artifacts(
    data: pd.DataFrame, 
    base_dir: str, 
    suggested_name: str, 
    table_name: str = "data"
) -> str:
    """
    Save `data` in a new timestamped folder inside `base_dir` using four methods:
    CSV, JSON, YAML, and SQLite. Files are named <method>_<suggested_name>.*
    
    Returns the created directory path.
    """
    # Make timestamped folder (UTC)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)  # use os as requested

    # 1) CSV
    csv_path = os.path.join(out_dir, f"csv_{suggested_name}.csv")
    data.to_csv(csv_path, index=False, encoding="utf-8")

    # 2) JSON (records)
    json_path = os.path.join(out_dir, f"json_{suggested_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    # 3) YAML (records)
    # Write YAML without requiring user to install extra packages:
    # We'll implement a tiny YAML emitter for simple records (no nested types).
    # For portability, we avoid external deps; format is readable and loadable by most YAML parsers.
    yaml_path = os.path.join(out_dir, f"yaml_{suggested_name}.yaml")
    records = data.to_dict(orient="records")
    def _to_simple_yaml(records):
        lines = []
        for rec in records:
            lines.append("-")
            for k, v in rec.items():
                # Use repr for float formatting; ensure dot decimal
                if isinstance(v, float):
                    v_str = repr(float(v))
                else:
                    v_str = str(v)
                lines.append(f"  {k}: {v_str}")
        return "\n".join(lines) + "\n"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(_to_simple_yaml(records))

    # 4) SQLite
    sqlite_path = os.path.join(out_dir, f"sqlite_{suggested_name}.sqlite")
    con = sqlite3.connect(sqlite_path)
    try:
        data.to_sql(table_name, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()

    return out_dir

# Run the saver function
output_dir = save_run_artifacts(df, "pytorch tests", "test")

# List the created files for convenience
created_files = sorted([str(Path(output_dir)/p) for p in os.listdir(output_dir)])

output_dir, created_files
