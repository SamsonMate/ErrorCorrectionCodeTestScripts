import numpy as np
import random
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm

# ========================================================
# Helper: introduce bit errors
# ========================================================
def introduce_bit_errors(encoded_bits: np.ndarray, p_error: float) -> np.ndarray:
    """Introduce independent bit flips into the codeword array."""
    if p_error <= 0:
        return encoded_bits.copy()
    flip_mask = np.random.rand(encoded_bits.size) < p_error
    corrupted = encoded_bits.copy()
    corrupted[flip_mask] ^= 1
    return corrupted

# ========================================================
# BCH encoding / decoding helper functions (pure Python)
# ========================================================
def gf2_poly_mul(a, b):
    """Multiply two polynomials over GF(2)."""
    result = [0]*(len(a)+len(b)-1)
    for i, coeff_a in enumerate(a):
        if coeff_a == 0:
            continue
        for j, coeff_b in enumerate(b):
            if coeff_b == 0:
                continue
            result[i+j] ^= 1
    return result

def gf2_poly_div(dividend, divisor):
    """Divide two polynomials over GF(2). Return quotient and remainder."""
    dividend = dividend.copy()
    deg_diff = len(dividend) - len(divisor)
    quotient = [0]*(deg_diff + 1) if deg_diff >= 0 else [0]
    while len(dividend) >= len(divisor):
        if dividend[0] == 1:
            quotient[len(dividend)-len(divisor)] = 1
            for i in range(len(divisor)):
                dividend[i] ^= divisor[i]
        dividend = dividend[1:]  # remove the leading term
    remainder = dividend
    return quotient, remainder

def bch_generator_poly(n, k):
    """
    Simple generator polynomial: g(x) = x^(n-k) + ... + 1
    Pure Python approximation for simulation purposes.
    """
    # For simulation, we construct a simple polynomial of degree n-k
    return [1] + [0]*(n-k-1) + [1]

def bch_encode(msg_bits: np.ndarray, n, k):
    """Encode message bits into BCH codeword."""
    g = bch_generator_poly(n, k)
    m_extended = np.concatenate([msg_bits, np.zeros(n - k, dtype=np.uint8)])
    _, remainder = gf2_poly_div(m_extended.tolist(), g)
    parity = np.array([0]*(n-k-len(remainder)) + remainder, dtype=np.uint8)
    codeword = np.concatenate([msg_bits, parity])
    return codeword

def bch_decode(codeword_bits: np.ndarray, n, k):
    """Decode BCH codeword. Returns message if successful, None if failure."""
    g = bch_generator_poly(n, k)
    _, remainder = gf2_poly_div(codeword_bits.tolist(), g)
    # If remainder all zeros → decode success
    if any(remainder):
        return None
    return codeword_bits[:k]

# ========================================================
# Perform one BCH encode/decode trial
# ========================================================
def run_single_trial(n_val: int, k_val: int, p_error: float) -> int:
    """Run a single BCH(n,k) encode-decode attempt."""
    if n_val <= 0 or k_val <= 0 or n_val <= k_val:
        return 1  # invalid parameters → automatic failure

    msg = np.random.randint(0, 2, k_val, dtype=np.uint8)
    codeword = bch_encode(msg, n_val, k_val)

    corrupted = introduce_bit_errors(codeword, p_error)

    decoded = bch_decode(corrupted, n_val, k_val)
    if decoded is None or not np.array_equal(decoded, msg):
        return 1  # failure
    return 0  # success

# ========================================================
# Run trials for a single block length
# ========================================================
def simulate_block_length(args):
    n_val, k_val, p_error, trials = args
    failures = 0
    start = time.time()
    for _ in range(trials):
        failures += run_single_trial(n_val, k_val, p_error)
    end = time.time()
    return failures / trials, end - start

# ========================================================
# Full parallel simulation with tqdm
# ========================================================
def run_simulation_parallel(block_lengths, parity_bits, p_error, trials_per_length):
    jobs = [(n_val, n_val - parity_bits, p_error, trials_per_length) for n_val in block_lengths]
    results = []
    total_jobs = len(jobs)

    with tqdm(total=total_jobs, desc="Simulating BCH(n,k)") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for res in pool.imap_unordered(simulate_block_length, jobs):
                results.append(res)
                pbar.update(1)

    error_rates, compute_times = zip(*results)
    return list(error_rates), list(compute_times)

# ========================================================
# Main entry
# ========================================================
if __name__ == "__main__":
    parity_bits = 32             # literal parity bits
    p_error = 0.1 #0.025 #1e-12              # bit error probability
    trials_per_length = 10000    # trials per block length

    # Valid BCH lengths: 2^m - 1 ≤ 255
    block_lengths = [15, 31, 63, 127, 255]

    print("Running BCH Simulation (Pure Python)...")
    error_rates, compute_times = run_simulation_parallel(
        block_lengths, parity_bits, p_error, trials_per_length
    )

    # ========================================================
    # Plotting
    # ========================================================
    plt.figure(figsize=(14, 4))
    plt.suptitle(
        f"BCH Simulation: parity={parity_bits}, bit error rate={p_error}"
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
