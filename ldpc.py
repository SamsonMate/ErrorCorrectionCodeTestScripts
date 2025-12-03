import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================================
# LDPC Code Definitions
# ================================
def generate_ldpc_matrix(n, k, weight=3):
    """Generate a random (n-k) x n sparse parity-check matrix H as a NumPy array."""
    m = n - k
    H = np.zeros((m, n), dtype=np.int8)
    for col in range(n):
        ones = np.random.choice(m, weight, replace=False)
        H[ones, col] = 1
    return H

# Example LDPC (64,32)
n = 64
k = 32
H = generate_ldpc_matrix(n, k, weight=3)
m = n - k

# Precompute neighbors
var_deg = [np.where(H[:, j])[0] for j in range(n)]
check_deg = [np.where(H[i, :])[0] for i in range(m)]

max_var_deg = max(len(v) for v in var_deg)
max_check_deg = max(len(c) for c in check_deg)

# Arrays for vectorization
var_nodes = -np.ones((n, max_var_deg), dtype=int)
for j, nodes in enumerate(var_deg):
    var_nodes[j, :len(nodes)] = nodes

check_nodes = -np.ones((m, max_check_deg), dtype=int)
for i, nodes in enumerate(check_deg):
    check_nodes[i, :len(nodes)] = nodes

var_mask = var_nodes >= 0
check_mask = check_nodes >= 0

# ================================
# Sum-Product Decoder (Vectorized)
# ================================
def ldpc_decode_sumprod(received, H, var_nodes, check_nodes, var_mask, check_mask, max_iter=10, p_error=0.02):
    B, n = received.shape
    m = H.shape[0]
    k = n - m

    llr = np.log((1-p_error)/(p_error)) * (1 - 2*received)  # shape (B,n)

    # Variable-to-check messages initialization
    M_vc = np.zeros((B, n, max_var_deg))
    for j in range(n):
        deg = np.sum(var_mask[j])
        M_vc[:, j, :deg] = llr[:, j][:, None]

    for _ in range(max_iter):
        # Check node update
        M_cv = np.zeros((B, m, max_check_deg))
        for i in range(m):
            deg = np.sum(check_mask[i])
            msgs = np.zeros((B, deg))
            for idx, var in enumerate(check_nodes[i, :deg]):
                msgs[:, idx] = M_vc[:, var, np.where(var_nodes[var]==i)[0][0]]
            tanh_vals = np.tanh(msgs/2)
            prod_all = np.prod(tanh_vals, axis=1, keepdims=True)
            for idx in range(deg):
                others = prod_all / tanh_vals[:, idx][:, None]
                M_cv[:, i, idx] = 2*np.arctanh(np.clip(others.flatten(), -0.999999, 0.999999))

        # Variable node update
        for j in range(n):
            deg = np.sum(var_mask[j])
            for idx, check in enumerate(var_nodes[j, :deg]):
                others = np.sum(np.delete(M_cv[:, check, :], np.where(check_nodes[check]==j)[0], axis=1), axis=1)
                M_vc[:, j, idx] = llr[:, j] + others

    llr_final = llr.copy()
    for j in range(n):
        deg = np.sum(var_mask[j])
        for idx, check in enumerate(var_nodes[j, :deg]):
            llr_final[:, j] += M_cv[:, check, idx]

    decoded = (llr_final < 0).astype(np.int8)
    return decoded[:, :k]

# ================================
# Parallel Trial Function
# ================================
def run_single_ldpc_trial(args):
    blk, p_error = args
    t0 = time.time()

    # Generate messages
    msgs = np.random.randint(0,2,(blk, k), dtype=np.int8)

    # Encode systematic: codeword = [msg | parity]
    codewords = np.zeros((blk, n), dtype=np.int8)
    codewords[:, :k] = msgs
    codewords[:, k:] = (msgs @ H[: , :k].T % 2).astype(np.int8)

    # BSC noise
    noise = (np.random.rand(blk, n) < p_error).astype(np.int8)
    received = codewords ^ noise

    # Decode
    decoded = ldpc_decode_sumprod(received, H, var_nodes, check_nodes, var_mask, check_mask, max_iter=10, p_error=p_error)
    total_err = np.sum(decoded != msgs)

    return total_err, time.time() - t0

# ================================
# Main Simulation Function
# ================================
def run_ldpc_simulation(block_lengths, p_error, trials_per_length):
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
            for err, t in tqdm(pool.imap_unordered(run_single_ldpc_trial, args),
                               total=trials_per_length,
                               desc=f"Block {blk}"):
                total_err += err
                total_time += t

        error_rates.append(total_err / (blk*k*trials_per_length))
        compute_times.append(total_time / trials_per_length)

    return np.array(error_rates), np.array(compute_times)

# ================================
# Run Simulation
# ================================
if __name__ == "__main__":
    p_error = 0.1 #0.025 #0.000000000001
    trials_per_length = 750
    block_lengths = [5, 10, 20, 50, 75, 100, 250, 500]

    error_rates, compute_times = run_ldpc_simulation(block_lengths, p_error, trials_per_length)

    # ================================
    # Plotting
    # ================================
    plt.figure(figsize=(14,4))
    plt.figure.__name__ = "LDPC Code Simulation Results at bit flip rate = {}".format(p_error)

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