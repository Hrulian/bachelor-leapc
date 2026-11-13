#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "log.csv"

DT = 0.01           # nominaler Regeltakt [s] - nur benutzt, wenn keine 't'-Spalte existiert
SKIP_SECONDS = 2.0  # erste N Sekunden aus allen Plots weglassen

def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"count", "u", "theta", "thetadot"}
    if not required.issubset(df.columns):
        print(f"Erwartete Spalten {sorted(required)} nicht gefunden. Gefunden: {list(df.columns)}")
        sys.exit(1)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))

    # Zeitachse [s]: echte 't'-Spalte bevorzugen, sonst aus count * DT rekonstruieren
    if "t" in df.columns:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
    else:
        df["t"] = df["count"] * DT

    # erste SKIP_SECONDS relativ zum Startzeitpunkt weglassen
    t0 = df["t"].iloc[0]
    df = df[df["t"] >= t0 + SKIP_SECONDS]

    plt.figure()
    plt.plot(df["t"], df["theta"], "-", color="C0")
    plt.xlabel("time [s]")
    plt.ylabel("theta")
    plt.grid(True)

    plt.figure()
    plt.plot(df["t"], df["thetadot"], "-", color="C1")
    plt.xlabel("time [s]")
    plt.ylabel("thetadot")
    plt.grid(True)

    plt.figure()
    plt.plot(df["t"], df["u"], "-", color="C2")
    plt.xlabel("time [s]")
    plt.ylabel("u")
    plt.grid(True)

    # separater Plot: Integralanteil (Ki * I) über die Zeit, falls vorhanden
    if "i_term" in df.columns:
        df["i_term"] = pd.to_numeric(df["i_term"], errors="coerce")
        df_i = df.dropna(subset=["i_term"])
        plt.figure()
        plt.plot(df_i["t"], df_i["i_term"], "-", color="C3")
        plt.xlabel("time [s]")
        plt.ylabel("i_term (Ki * I)")
        plt.title("Integralanteil ueber die Zeit")
        plt.grid(True)

    # zusätzlicher Plot: theta und thetadot mit eigener y-Achse
    fig, ax1 = plt.subplots()
    ax1.plot(df["t"], df["theta"], color="C0", label="theta")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("theta", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["thetadot"], color="C1", label="thetadot")
    ax2.set_ylabel("thetadot", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    # gemeinsame Legende (falls gewünscht)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
