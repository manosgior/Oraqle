"""
Build the results/ inputs for plot_mcm_qec_thesis.py from the regenerated
synchronization-artifact CSVs (updated herqules noise).

Combined single CSV = the 4 tradeoff files tagged with system/noise_model:
  mcm_tradeoff_mcm_ibm.csv            -> MCMit    / current
  mcm_tradeoff_herqules_ibm.csv       -> HERQULES / current
  mcm_tradeoff_mcm_futuristic_2.csv   -> MCMit    / futuristic
  mcm_tradeoff_herqules_futuristic_2.csv -> HERQULES / futuristic
"""
import csv, os, shutil

ART = "synchronization-artifact"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

SOURCES = [
    ("mcm_tradeoff_mcm_ibm.csv",            "MCMit",    "current"),
    ("mcm_tradeoff_herqules_ibm.csv",       "HERQULES", "current"),
    ("mcm_tradeoff_mcm_futuristic_2.csv",   "MCMit",    "futuristic"),
    ("mcm_tradeoff_herqules_futuristic_2.csv", "HERQULES", "futuristic"),
]

header = ["distance", "measure_latency_ns", "measure_error",
          "logical_error_rate", "system", "noise_model"]
rows = []
for fname, system, noise in SOURCES:
    path = os.path.join(ART, fname)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append([r["distance"], r["measure_latency_ns"], r["measure_error"],
                         r["logical_error_rate"], system, noise])
    print(f"  + {fname:42} -> {system:8} / {noise:10} ({sum(1 for _ in open(path))-1} rows)")

combined = os.path.join(OUT, "mcm_latency_error_ler.csv")
with open(combined, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"wrote {combined} ({len(rows)} rows)")

# Panel (c) patches + (d) error inputs: copy the artifact versions verbatim
# (mcm_patch_dist.csv uses the 'total_patches' header the script expects).
for fname in ["mcm_patch_dist.csv", "mcm_error_latency_ler.csv"]:
    shutil.copy(os.path.join(ART, fname), os.path.join(OUT, fname))
    print(f"copied {fname} -> {OUT}/")
