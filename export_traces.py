import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

def export_averaged_traces(
    h5_file_path: str,
    output_png_path: str,
    num_points: int,
    indices_0: np.ndarray,
    indices_1: np.ndarray
):
    dFreq = -127.282e6
    sampling_rate = 500e6
    dt = 1.0 / sampling_rate

    # Load the data and average
    with h5py.File(h5_file_path, 'r') as f:
        X = f['X_train']
        indices_0 = np.sort(indices_0)
        indices_1 = np.sort(indices_1)
        
        traces_0 = X[indices_0]
        traces_1 = X[indices_1]
        
    trace_0 = np.mean(traces_0, axis=0)
    trace_1 = np.mean(traces_1, axis=0)

    def demodulate(trace):
        trace_length = trace.shape[0]
        vTime = np.arange(trace_length) * dt
        vCos = np.cos(2 * np.pi * vTime * dFreq)
        vSin = np.sin(2 * np.pi * vTime * dFreq)
        
        DataI = trace[:, 0]
        DataQ = trace[:, 1]
        
        DataI = DataI - np.mean(DataI)
        DataQ = DataQ - np.mean(DataQ)
        corr_factor = np.std(DataI) / np.std(DataQ)
        DataQ = DataQ * corr_factor

        i_mixed = DataI * vCos + DataQ * vSin
        q_mixed = DataQ * vCos - DataI * vSin
        
        sos = butter(3, 10e6, btype='low', fs=sampling_rate, output='sos')
        i_filtered = sosfilt(sos, i_mixed)
        q_filtered = sosfilt(sos, q_mixed)
        
        return i_filtered, q_filtered

    t0_I_full, t0_Q_full = demodulate(trace_0)
    t1_I_full, t1_Q_full = demodulate(trace_1)

    step = max(1, 500 // num_points)
    
    t0_I, t0_Q = t0_I_full[:500:step], t0_Q_full[:500:step]
    t1_I, t1_Q = t1_I_full[:500:step], t1_Q_full[:500:step]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(t0_I, t0_Q, marker='o', linestyle='-', color='#4C72B0', label='0', markersize=8, linewidth=2)
    ax.plot(t1_I, t1_Q, marker='o', linestyle='--', color='#55A868', label='1', markersize=8, linewidth=2)

    # Add text annotations cleanly above the points
    ax.annotate('t=0', (t0_I[0], t0_Q[0]), xytext=(0, 10), textcoords='offset points', fontsize=16, ha='center', va='bottom')
    ax.annotate('t=0', (t1_I[0], t1_Q[0]), xytext=(0, 10), textcoords='offset points', fontsize=16, ha='center', va='bottom')
    ax.annotate('t=1μs', (t0_I[-1], t0_Q[-1]), xytext=(0, 10), textcoords='offset points', fontsize=16, ha='center', va='bottom')
    ax.annotate('t=1μs', (t1_I[-1], t1_Q[-1]), xytext=(0, 10), textcoords='offset points', fontsize=16, ha='center', va='bottom')

    ax.set_xlabel('In-phase (I)', fontsize=18)
    ax.set_ylabel('Quadrature (Q)', fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=16, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to {output_png_path} (Averaged {len(indices_0)} traces)")
    
    if len(indices_0) == 5000:
        print("\n--- EXACT 15 POINTS PLOTTED FOR THE 5000-TRACE AVERAGE ---")
        print("State 0 (I, Q):")
        for i in range(len(t0_I)):
            print(f"  Point {i+1}: ({t0_I[i]:.6f}, {t0_Q[i]:.6f})")
        print("\nState 1 (I, Q):")
        for i in range(len(t1_I)):
            print(f"  Point {i+1}: ({t1_I[i]:.6f}, {t1_Q[i]:.6f})")
        print("----------------------------------------------------------\n")

if __name__ == "__main__":
    data_path = "/home/manosgior/Documents/GitHub/KLiNQ/qubit_readout_klinq/data/five_qubit_data/DRaw_C_Tr_v0-001"
    print("Loading data to find indices...")
    with h5py.File(data_path, 'r') as f:
        y = f['y_train'][:]
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 16)[0]
    
    batch_sizes = [50, 200, 500, 1000, 5000]
    for i, size in enumerate(batch_sizes):
        out_file = f"readout_traces_figure_{i+1}.png"
        np.random.seed(42 + i)
        chosen_idx_0 = np.random.choice(idx_0, size, replace=False)
        chosen_idx_1 = np.random.choice(idx_1, size, replace=False)
        
        export_averaged_traces(
            h5_file_path=data_path,
            output_png_path=out_file,
            num_points=15,
            indices_0=chosen_idx_0,
            indices_1=chosen_idx_1
        )
