import os
import glob
import csv
import re

src_glob = "./data/pwm*.csv"
out_dir = "./data/converted"
os.makedirs(out_dir, exist_ok=True)

for path in glob.glob(src_glob):
    name = os.path.basename(path)
    m = re.match(r"pwm-rev-(\d+)\.csv$", name)
    if m:
        pwm = -int(m.group(1))
    else:
        m2 = re.match(r"pwm-(\d+)\.csv$", name)
        if m2:
            pwm = int(m2.group(1))
        else:
            print("Überspringe unbekannte Datei:", name)
            continue

    out_path = os.path.join(out_dir, name)
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                time_s = float(parts[0])
                x = float(parts[1])
            except ValueError:
                # überspringe Header oder nicht-numerische Zeilen
                continue
            # Zeit jetzt in Sekunden (float), nicht in Mikrosekunden
            rows.append([i, pwm, time_s, x])

    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        for r in rows:
            writer.writerow(r)

    print("Konvertiert:", name, "->", os.path.relpath(out_path))