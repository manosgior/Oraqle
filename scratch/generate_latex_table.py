
data = [
    {"name": "Baseline", "q": [96.6960, 73.0021, 89.4028, 93.6284, 95.7660], "gm5": 89.2366, "gm4": 93.8306},
    {"name": "arXiv FNN", "q": [96.7350, 73.1950, 89.4413, 93.7925, 95.6450], "gm5": 89.3072, "gm4": 93.8616},
    {"name": "CNN", "q": [96.8255, 74.0430, 94.1632, 94.5126, 96.9806], "gm5": 90.8460, "gm4": 95.6118},
    {"name": "FNN", "q": [96.7788, 74.8675, 93.7188, 94.5025, 96.9375], "gm5": 90.9425, "gm4": 95.4741},
    {"name": "HERQULES", "q": [96.7837, 74.4087, 92.7925, 94.1750, 96.5888], "gm5": 90.5233, "gm4": 95.0702},
    {"name": "Transformer", "q": [96.2075, 74.5175, 93.6275, 93.8362, 96.7588], "gm5": 90.5704, "gm4": 95.0973},
    {"name": "KLiNQ (Student)", "q": [96.8700, 74.3337, 93.8000, 94.4838, 96.9100], "gm5": 90.8366, "gm4": 95.5058},
]

# Convert all to 0-1 range
for d in data:
    d["q"] = [x / 100.0 for x in d["q"]]
    d["gm5"] /= 100.0
    d["gm4"] /= 100.0

print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Discriminator Accuracies for Trace Length 500}")
print("\\begin{tabular}{l|ccccc|c|c}")
print("\\hline")
print("Model & Q0 & Q1 & Q2 & Q3 & Q4 & GM(5) & GM(4) \\\\")
print("\\hline")
for d in data:
    qs = " & ".join([f"{x:.4f}" for x in d["q"]])
    print(f"{d['name']} & {qs} & {d['gm5']:.4f} & {d['gm4']:.4f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")
