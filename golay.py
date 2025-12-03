import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

DT = np.int8

# ================================
# Golay (23,12) Definitions
# ================================
H = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0],
    [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],
    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    [1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0],
    [1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
    [1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
    [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
], dtype=DT)

# Precompute syndrome table
syndrome_table = {}
def populate_syndrome_table():
    zero23 = np.zeros(23, dtype=DT)
    zero11 = tuple(np.zeros(11, dtype=DT))
    syndrome_table[zero11] = zero23

    # weight-1
    for i in range(23):
        e = np.zeros(23, dtype=DT); e[i]=1
        syndrome_table[tuple((H @ e) % 2)] = e.copy()

    # weight-2
    for i in range(23):
        for j in range(i+1,23):
            e = np.zeros(23, dtype=DT); e[i]=1; e[j]=1
            syndrome_table[tuple((H @ e) % 2)] = e.copy()

    # weight-3
    for i in range(23):
        for j in range(i+1,23):
            for k in range(j+1,23):
                e = np.zeros(23, dtype=DT); e[i]=1; e[j]=1; e[k]=1
                syndrome_table[tuple((H @ e) % 2)] = e.copy()

populate_syndrome_table()

# ================================
# Vectorized Golay Encode / Decode
# ================================
def encode_golay_vec(msgs):
    tmp = np.hstack([msgs, np.zeros((msgs.shape[0],11), dtype=DT)])
    s = (tmp @ H.T) % 2
    return (tmp ^ np.hstack([np.zeros((msgs.shape[0],12),dtype=DT), s])) % 2

def decode_golay_vec(received):
    synd = (received @ H.T) % 2
    # Vectorized correction
    corrections = np.zeros_like(received)
    # Map unique syndromes to corrections
    unique_synds, inv_idx = np.unique(synd, axis=0, return_inverse=True)
    for i, s in enumerate(unique_synds):
        correction = syndrome_table.get(tuple(s), np.zeros(23, dtype=DT))
        corrections[inv_idx == i] = correction
    return (received ^ corrections)[:, :12]

# ================================
# Single Trial (for Parallel)
# ================================
def run_single_trial(args):
    blk, p_error = args
    t0 = time.time()

    msgs = np.random.randint(0,2,(blk,12), dtype=DT)
    codewords = encode_golay_vec(msgs)
    noise = (np.random.rand(blk,23) < p_error).astype(DT)
    received = codewords ^ noise
    decoded = decode_golay_vec(received)

    total_err = np.sum(decoded != msgs)
    return total_err, time.time() - t0

# ================================
# Main Simulation with Parallelism
# ================================
def run_simulation_parallel(block_lengths, p_error, trials_per_length):
    error_rates = []
    compute_times = []

    num_workers = cpu_count()
    print(f"Using {num_workers} parallel workers.")

    for blk in block_lengths:
        print(f"\nRunning block length = {blk}")
        args = [(blk, p_error)] * trials_per_length

        total_err = 0
        total_time = 0

        with Pool(num_workers) as pool:
            for err, t in tqdm(pool.imap_unordered(run_single_trial, args),
                        total=trials_per_length,
                        desc=f"Block {blk}"):
                total_err += err
                total_time += t

        error_rates.append(total_err / (blk*12*trials_per_length))
        compute_times.append(total_time / trials_per_length)

    return np.array(error_rates), np.array(compute_times)

# ================================
# Run Simulation
# ================================
if __name__ == "__main__":
    p_error = 0.1 #0.025 #0.000000000001
    trials_per_length = 10000
    block_lengths = [10, 25, 50, 75, 100, 200, 300, 500, 1000]

    error_rates, compute_times = run_simulation_parallel(block_lengths, p_error, trials_per_length)

    # ================================
    # Plotting
    # ================================
    plt.figure(figsize=(14,4))
    plt.figure.__name__ = "Golay Code Simulation Results at bit flip rate = {}".format(p_error)

    plt.subplot(1,3,1)
    plt.plot(block_lengths, error_rates, marker='o')
    plt.xlabel("Block Length")
    plt.ylabel("% of Messages Lost")
    plt.title("Block Length vs % of Messages Lost")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.subplot(1,3,2)
    plt.plot(block_lengths, compute_times, marker='o', color='orange')
    plt.xlabel("Block Length")
    plt.ylabel("Compute Time (s)")
    plt.title("Block Length vs Computation Time")
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

    plt.subplot(1,3,3)
    plt.plot(compute_times, error_rates, marker='o', color='green')
    plt.xlabel("Compute Time (s)")
    plt.ylabel("% of Messages Lost")
    plt.title("Computation Time vs % of Messages Lost")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.show()
