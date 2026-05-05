import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_iq_clusters(
    h5_file_path: str,
    output_png_path: str,
    n_shots: int = 500
):
    dFreq = -127.282e6
    sampling_rate = 500e6
    dt = 1.0 / sampling_rate

    print("Loading data...")
    with h5py.File(h5_file_path, 'r') as f:
        y = f['y_train'][:]
        X = f['X_train']
        
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 16)[0]
        
        np.random.seed(42)
        chosen_idx_0 = np.sort(np.random.choice(idx_0, n_shots, replace=False))
        chosen_idx_1 = np.sort(np.random.choice(idx_1, n_shots, replace=False))
        
        traces_0 = X[chosen_idx_0]
        traces_1 = X[chosen_idx_1]

    trace_length = traces_0.shape[1]
    vTime = np.arange(trace_length) * dt
    vCos = np.cos(2 * np.pi * vTime * dFreq)
    vSin = np.sin(2 * np.pi * vTime * dFreq)
    
    def process_traces(traces):
        DataI = traces[:, :, 0]
        DataQ = traces[:, :, 1]
        
        # Normalize per trace
        DataI = DataI - np.mean(DataI, axis=1, keepdims=True)
        DataQ = DataQ - np.mean(DataQ, axis=1, keepdims=True)
        corr_factor = np.std(DataI, axis=1, keepdims=True) / np.std(DataQ, axis=1, keepdims=True)
        DataQ = DataQ * corr_factor
        
        # Mix with LO
        i_mixed = DataI * vCos + DataQ * vSin
        q_mixed = DataQ * vCos - DataI * vSin
        
        # Average the whole trace into a single value
        return np.mean(i_mixed, axis=1), np.mean(q_mixed, axis=1)

    print("Demodulating and integrating traces...")
    i_0, q_0 = process_traces(traces_0)
    i_1, q_1 = process_traces(traces_1)

    # Calculate cluster centers
    c0_i, c0_q = np.mean(i_0), np.mean(q_0)
    c1_i, c1_q = np.mean(i_1), np.mean(q_1)

    # Calculate decision boundary (perpendicular bisector)
    mid_i = (c0_i + c1_i) / 2.0
    mid_q = (c0_q + c1_q) / 2.0
    
    slope = (c1_q - c0_q) / (c1_i - c0_i)
    perp_slope = -1.0 / slope

    # Generate points for the boundary line
    i_min = min(np.min(i_0), np.min(i_1))
    i_max = max(np.max(i_0), np.max(i_1))
    # Extend line a bit past the clusters
    i_range = i_max - i_min
    i_vals = np.array([i_min - 0.2*i_range, i_max + 0.2*i_range])
    q_vals = perp_slope * (i_vals - mid_i) + mid_q

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Scatter points
    ax.scatter(i_0, q_0, color='#4C72B0', alpha=0.6, label='0', edgecolors='none', s=30)
    ax.scatter(i_1, q_1, color='#55A868', alpha=0.6, label='1', edgecolors='none', s=30)
    
    # Plot decision boundary
    ax.plot(i_vals, q_vals, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    
    # Plot centers
    ax.scatter([c0_i], [c0_q], color='black', marker='X', s=100)
    ax.scatter([c1_i], [c1_q], color='black', marker='X', s=100)

    # Set limits based on data so boundary line doesn't skew the plot
    q_min = min(np.min(q_0), np.min(q_1))
    q_max = max(np.max(q_0), np.max(q_1))
    q_range = q_max - q_min
    ax.set_xlim(i_min - 0.2*i_range, i_max + 0.2*i_range)
    ax.set_ylim(q_min - 0.2*q_range, q_max + 0.2*q_range)

    ax.set_xlabel('In-phase (I)', fontsize=16)
    ax.set_ylabel('Quadrature (Q)', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to {output_png_path}")

if __name__ == "__main__":
    data_path = "/home/manosgior/Documents/GitHub/KLiNQ/qubit_readout_klinq/data/five_qubit_data/DRaw_C_Tr_v0-001"
    
    # Generate variations with 30, 200, and 1000 traces so the user can choose
    for shots in [30, 200, 600]:
        out_file = f"readout_clusters_{shots}_shots.png"
        plot_iq_clusters(data_path, out_file, n_shots=shots)

