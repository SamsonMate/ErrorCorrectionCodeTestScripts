import numpy as np
import random
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import reedsolo
import matplotlib.ticker as mtick
from tqdm import tqdm

# ========================================================
# Helper: introduce symbol errors
# ========================================================
def introduce_symbol_errors(encoded: bytes, num_errors: int) -> bytearray:
    if num_errors <= 0:
        return bytearray(encoded)

    encoded_len = len(encoded)
    num_errors = min(int(num_errors), encoded_len)

    positions = np.random.choice(encoded_len, size=num_errors, replace=False)
    corrupted = bytearray(encoded)

    for p in positions:
        corrupted[p] ^= random.randint(1, 255)

    return corrupted

# ========================================================
# Perform one RS encode/decode trial
# ========================================================
def run_single_trial(N_val: int, K_val: int, p_error: float) -> int:
    """
    Run a single RS(N, K) encode-decode attempt.
    """

    # Generate RS codec
    nsym = N_val - K_val
    if nsym <= 0:
        return 1  # no parity symbols → automatic failure

    RS = reedsolo.RSCodec(nsym)

    # random K-byte message
    msg = bytearray(np.random.randint(0, 256, K_val, dtype=np.uint8))

    encoded = RS.encode(msg)

    # simulate errors across N symbols
    num_errors = int(np.sum(np.random.rand(N_val) < p_error))
    corrupted = introduce_symbol_errors(encoded, num_errors)

    try:
        RS.decode(corrupted)
        return 0
    except reedsolo.ReedSolomonError:
        return 1
    except Exception:
        return 1

# ========================================================
# Run trials for a single block length
# ========================================================
def simulate_block_length(args):
    N_val, parity_bits, p_error, trials = args

    # Compute K from the new rule
    K_val = N_val - parity_bits

    # Basic validation
    if N_val <= 0 or N_val > 255:
        return 1.0, 0.0   # reedsolo requires N ≤ 255
    if K_val <= 0:
        return 1.0, 0.0   # cannot encode a <=0-sized message

    failures = 0
    start = time.time()

    for _ in range(trials):
        failures += run_single_trial(N_val, K_val, p_error)

    end = time.time()

    return failures / trials, end - start

# ========================================================
# Full parallel simulation with tqdm progress bar
# ========================================================
def run_simulation_parallel(block_lengths, parity_bits, p_error, trials_per_length):
    jobs = []
    invalid_indices = {}

    for idx, N_val in enumerate(block_lengths):

        K_val = N_val - parity_bits
        if N_val <= 0 or N_val > 255 or K_val <= 0:
            invalid_indices[idx] = N_val
        else:
            jobs.append((N_val, parity_bits, p_error, trials_per_length))

    results = []
    total_jobs = len(jobs)

    with tqdm(total=total_jobs, desc="Simulating RS(N,K)") as pbar:
        if jobs:
            with mp.Pool(mp.cpu_count()) as pool:
                for res in pool.imap_unordered(simulate_block_length, jobs):
                    results.append(res)
                    pbar.update(1)

    error_rates = []
    compute_times = []
    res_iter = iter(results)

    for idx, N_val in enumerate(block_lengths):
        if idx in invalid_indices:
            error_rates.append(1.0)
            compute_times.append(0.0)
        else:
            er, ct = next(res_iter)
            error_rates.append(er)
            compute_times.append(ct)

    return error_rates, compute_times

# ========================================================
# Main entry
# ========================================================
if __name__ == "__main__":

    # ====================================================
    # Simulation variables
    # ====================================================
    parity_bits = 8             # number of RS parity symbols
    p_error = 1e-12 #0.025 #              # symbol-level error probability
    trials_per_length = 2500      # RS encode/decode trials per N
    block_lengths = [10, 25, 50, 75, 100, 175, 255] # N values

    print("Running Reed–Solomon Simulation...")
    error_rates, compute_times = run_simulation_parallel(
        block_lengths, parity_bits, p_error, trials_per_length
    )

    # ========================================================
    # Plotting
    # ========================================================
    plt.figure(figsize=(14, 4))
    plt.suptitle(
        f"Reed–Solomon Simulation: parity={parity_bits}, symbol error rate={p_error}"
    )

    # Plot 1: Error Rate vs N
    plt.subplot(1, 3, 1)
    plt.plot(block_lengths, error_rates, marker='o')
    plt.xlabel("Codeword Length N")
    plt.ylabel("% of Messages Lost")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.title("N vs % of Messages Lost")

    # Plot 2: Compute Time vs N
    plt.subplot(1, 3, 2)
    plt.plot(block_lengths, compute_times, marker='o')
    plt.xlabel("Codeword Length N")
    plt.ylabel("Compute Time (s)")
    plt.title("N vs Compute Time")
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

    # Plot 3: Error Rate vs Compute Time
    plt.subplot(1, 3, 3)
    plt.scatter(compute_times, error_rates, marker='o')
    plt.xlabel("Compute Time (s)")
    plt.ylabel("% of Messages Lost")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.title("Compute Time vs Error Rate")

    plt.tight_layout()
    plt.show()
