#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "fix5.csv"

def load_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"Fehler beim Einlesen von {path}: {e}")
        sys.exit(1)

    # Spaltennamen säubern
    df.columns = [c.strip().lower() for c in df.columns]

    # Aliasse vereinheitlichen
    if "time" in df.columns and "t" not in df.columns:
        df.rename(columns={"time": "t"}, inplace=True)
    if "x" in df.columns and "m" not in df.columns:
        df.rename(columns={"x": "m"}, inplace=True)

    required = {"t", "m", "v"}
    if not required.issubset(df.columns):
        print(f"Erwartete Spalten {sorted(required)} nicht vollständig vorhanden. "
              f"Gefunden: {list(df.columns)}")
        sys.exit(1)

    # Zahlen sicherstellen (fehlerhafte Einträge werden zu NaN)
    for col in ["t", "m", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "m", "v"])

    return df

df = load_csv(CSV_PATH)

# Plot 1: x(m) über Zeit
plt.figure()
plt.plot(df["t"], df["m"])
plt.xlabel("Zeit t [s]")
plt.ylabel("Position x [m]")
plt.grid(True)

# Plot 2: v über Zeit
plt.figure()
plt.plot(df["t"], df["v"])
plt.xlabel("Zeit t [s]")
plt.ylabel("Geschwindigkeit v [m/s]")
plt.grid(True)

plt.show()
