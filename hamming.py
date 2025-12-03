import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math

# ======================================================
# Build Hamming (n,k) Code
# ======================================================
def build_hamming_code(m):
    """
    Hamming code: n = 2^m - 1, k = n - m
    Returns (n, k, H, G)
    """
    n = (1 << m) - 1
    k = n - m

    H = np.zeros((m, n), dtype=np.int8)

    # Fill H with binary column indices 1..n
    for col in range(1, n + 1):
        bin_digits = np.array(list(np.binary_repr(col, width=m)), dtype=np.int8)
        H[:, col - 1] = bin_digits[::-1]   # LSB at top (common convention)

    # Generator matrix G in systematic form [I_k | P]
    P = H[:, :k].T % 2
    G = np.hstack((np.eye(k, dtype=np.int8), P))

    return n, k, H, G


# ======================================================
# Encode Hamming
# ======================================================
def encode_hamming(msgs, G):
    return (msgs @ G) % 2


# ======================================================
# Decode Hamming
# ======================================================
def decode_hamming(received, H):
    B, n = received.shape
    m = H.shape[0]
    k = n - m

    synd = (received @ H.T) % 2
    decoded = received.copy()

    for i in range(B):
        s_val = int("".join(str(x) for x in synd[i][::-1]), 2)
        if s_val != 0:
            decoded[i, s_val - 1] ^= 1  # flip bit

    return decoded[:, :k]


# ======================================================
# Single trial for multiprocessing
# ======================================================
def run_single_trial(args):
    blk, p_error, H, G, n, k = args
    t0 = time.time()

    msgs = np.random.randint(0, 2, (blk, k), dtype=np.int8)
    codewords = encode_hamming(msgs, G)

    noise = (np.random.rand(blk, n) < p_error).astype(np.int8)
    received = codewords ^ noise

    decoded = decode_hamming(received, H)

    total_err = np.sum(decoded != msgs)
    return total_err, time.time() - t0


# ======================================================
# Main simulation wrapper (same as BCH version)
# ======================================================
def run_sim(block_lengths, p_error, trials):
    valid_ns = []
    valid_ms = []
    code_data = {}

    # determine valid Hamming lengths
    for n in block_lengths:
        m = math.log2(n + 1)
        if m.is_integer():
            m = int(m)
            valid_ns.append(n)
            valid_ms.append(m)

            # build code
            n2, k, H, G = build_hamming_code(m)
            code_data[n] = (m, n2, k, H, G)

        else:
            print(f"❌ Skipping invalid Hamming length n={n}. Must be 2^m - 1.")

    if not valid_ns:
        print("No valid Hamming block lengths found.")
        return np.array([]), np.array([]), [], []

    error_rates = []
    compute_times = []
    num_workers = cpu_count()
    print(f"\nUsing {num_workers} worker processes.\n")

    # Run simulation for each valid Hamming code
    for n in valid_ns:
        m, n2, k, H, G = code_data[n]
        print(f"▶ Running Hamming({n2},{k})  [m={m}]")

        args = [(1, p_error, H, G, n2, k)] * trials

        total_err = 0
        total_t = 0

        with Pool(num_workers) as pool:
            for err, t in tqdm(pool.imap_unordered(run_single_trial, args),
                               total=trials, desc=f"n={n}"):
                total_err += err
                total_t += t

        # Bit error rate (BER)
        error_rates.append(total_err / (trials * k))

        # Avg compute time per message
        compute_times.append(total_t / trials)

    return np.array(error_rates), np.array(compute_times), valid_ns, valid_ms


# ======================================================
# Main + plotting
# ======================================================
if __name__ == "__main__":
    block_lengths = [7, 15, 31, 63, 127, 1023]
    p_error = 0.1 #0.000000000001 #0.025
    trials = 10000

    error_rates, compute_times, valid_ns, valid_ms = run_sim(
        block_lengths, p_error, trials
    )

    if len(valid_ns) == 0:
        exit()

    # PLOTS (exact same as BCH version)
    plt.figure(figsize=(14, 4))
    plt.figure.__name__ = "Hamming Code Simulation Results"

    # 1 — Error rate vs block length
    plt.subplot(1, 3, 1)
    plt.plot(valid_ns, error_rates, marker="o")
    plt.xlabel("Block Length (n)")
    plt.ylabel("% of Messages Lost")
    plt.title("Block Length vs % Lost")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # 2 — Compute time
    plt.subplot(1, 3, 2)
    plt.plot(valid_ns, compute_times, marker="o", color="orange")
    plt.xlabel("Block Length (n)")
    plt.ylabel("Compute Time (s)")
    plt.title("Block Length vs Compute Time")
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

    # 3 — Compute time vs error rate
    plt.subplot(1, 3, 3)
    plt.plot(compute_times, error_rates, marker="o", color="green")
    plt.xlabel("Compute Time (s)")
    plt.ylabel("% of Messages Lost")
    plt.title("Compute Time vs % Lost")
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.show()
