
# Model Name Mapping
name_map = {
    "Baseline": "Linear",
    "arXiv FNN": "QubiCML",
    "CNN": "MCMit CNN",
    "FNN": "Baseline FNN",
    "HERQULES": "HERQULES",
    "Transformer": "MCMit Transformer",
    "KLiNQ (Student)": "KLiNQ"
}

# Table 1 Data (Trace Length 500)
table1_data = [
    {"name": "Baseline", "q": [0.9670, 0.7300, 0.8940, 0.9363, 0.9577], "gm5": 0.8924, "gm4": 0.9383},
    {"name": "arXiv FNN", "q": [0.9674, 0.7319, 0.8944, 0.9379, 0.9564], "gm5": 0.8931, "gm4": 0.9386},
    {"name": "CNN", "q": [0.9683, 0.7404, 0.9416, 0.9451, 0.9698], "gm5": 0.9085, "gm4": 0.9561},
    {"name": "FNN", "q": [0.9678, 0.7487, 0.9372, 0.9450, 0.9694], "gm5": 0.9094, "gm4": 0.9547},
    {"name": "HERQULES", "q": [0.9678, 0.7441, 0.9279, 0.9417, 0.9659], "gm5": 0.9052, "gm4": 0.9507},
    {"name": "Transformer", "q": [0.9621, 0.7452, 0.9363, 0.9384, 0.9676], "gm5": 0.9057, "gm4": 0.9510},
    {"name": "KLiNQ (Student)", "q": [0.9687, 74.3337/100.0, 93.8000/100.0, 94.4838/100.0, 96.9100/100.0], "gm5": 0.9084, "gm4": 0.9551},
]

# Table 2 Data (Trace Lengths)
table2_data = {
    "Baseline": [0.7388, 0.8635, 0.8883, 0.8930, 0.8924],
    "arXiv FNN": [0.7381, 0.8644, 0.8889, 0.8933, 0.8931],
    "CNN": [0.6696, 0.8568, 0.8944, 0.9045, 0.9085],
    "FNN": [0.7666, 0.8772, 0.8994, 0.9063, 0.9094],
    "HERQULES": [0.7661, 0.8751, 0.8965, 0.9037, 0.9052],
    "Transformer": [0.7644, 0.8763, 0.8974, 0.9051, 0.9057],
    "KLiNQ (Student)": [0.7673, 0.8769, 0.8984, 0.9054, 0.9084]
}

def print_table1():
    print("% Table 1: Accuracies for Length 500")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Discriminator Accuracies for Trace Length 500 (1 $\\mu$s)}")
    print("\\begin{tabular}{l|ccccc|c|c}")
    print("\\hline")
    print("Model & Q0 & Q1 & Q2 & Q3 & Q4 & GM(5) & GM(4) \\\\")
    print("\\hline")
    for d in table1_data:
        new_name = name_map[d["name"]]
        qs = " & ".join([f"{x:.3f}" for x in d["q"]])
        print(f"{new_name} & {qs} & {d['gm5']:.3f} & {d['gm4']:.3f} \\\\")
        print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

def print_table2():
    print("\n% Table 2: GM Fidelity vs Length")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Geometric Mean Fidelity (5 Qubits) vs. Trace Length}")
    print("\\begin{tabular}{l|ccccc}")
    print("\\hline")
    print("Model & 100 & 200 & 300 & 400 & 500 \\\\")
    print("\\hline")
    # Order to match renames
    order = ["Baseline", "arXiv FNN", "CNN", "FNN", "HERQULES", "Transformer", "KLiNQ (Student)"]
    for old_name in order:
        new_name = name_map[old_name]
        values = table2_data[old_name]
        val_str = " & ".join([f"{v:.3f}" for v in values])
        print(f"{new_name} & {val_str} \\\\")
        print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

print_table1()
print_table2()
