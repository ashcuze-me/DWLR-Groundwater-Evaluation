
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    ROOT / "data" / "raw" / "team_formatted_data"
    / "gwl_ap_1991_2020.csv"
)

print("Inspecting original Andhra Pradesh dataset...")
print(f"File: {SOURCE_FILE}")

df = pd.read_csv(SOURCE_FILE, nrows=10, low_memory=False)

print("\nORIGINAL COLUMN NAMES")
for index, column in enumerate(df.columns, start=1):
    print(f"{index}. {column}")

print("\nTOTAL ORIGINAL COLUMNS:", len(df.columns))

print("\nFIRST 10 ORIGINAL RECORDS")
print(df.to_string(index=False))

print("\nDATA TYPES")
print(df.dtypes.to_string())
