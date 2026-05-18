
data = {
    "Baseline": [0.7388, 0.8635, 0.8883, 0.8930, 0.8924],
    "arXiv FNN": [0.7381, 0.8644, 0.8889, 0.8933, 0.8931],
    "CNN": [0.6696, 0.8568, 0.8944, 0.9045, 0.9085],
    "FNN": [0.7666, 0.8772, 0.8994, 0.9063, 0.9094],
    "HERQULES": [0.7661, 0.8751, 0.8965, 0.9037, 0.9052],
    "Transformer": [0.7644, 0.8763, 0.8974, 0.9051, 0.9057],
    "KLiNQ (Student)": [0.7673, 0.8769, 0.8984, 0.9054, 0.9084]
}

lengths = [100, 200, 300, 400, 500]

print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Geometric Mean Fidelity (5 Qubits) vs. Trace Length}")
print("\\begin{tabular}{l|ccccc}")
print("\\hline")
print("Model & 100 & 200 & 300 & 400 & 500 \\\\")
print("\\hline")
for name, values in data.items():
    val_str = " & ".join([f"{v:.4f}" for v in values])
    print(f"{name} & {val_str} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")
