"""Phase 2: Strong/weak scaling benchmark for the parallel computing lab.

Uses only Python's standard library and the existing harness.py module.
Run from the directory containing harness.py:
    python3 task2_scaling.py
"""

import math
import multiprocessing as mp
import statistics
import time

from harness import emit_verify_tag, run_benchmark


# ---------------------------------------------------------------------------
# Configuration from the lab handout and the student's Phase 0 measurements
# ---------------------------------------------------------------------------
STUDENT_ID = 240103149
P_LOG = mp.cpu_count()

N_STRONG = 1_600_000
N0_WEAK = 250_000
SERIAL_PREFIX_FRACTION = 0.05
SERIAL_TRANSCENDENTAL_PAIRS = 6  # 6 * (sin + sqrt) = 12 operations / element
PARALLEL_ITERATIONS = 35

WARMUP_RUNS = 2
TIMED_RUNS = 7


# ---------------------------------------------------------------------------
# Stage A: serial work in the parent process
# ---------------------------------------------------------------------------
def serial_stage(n):
    """Process the first 5% of n elements using 12 transcendental ops each."""
    prefix_n = max(1, int(n * SERIAL_PREFIX_FRACTION))
    total = 0.0

    for i in range(prefix_n):
        x = (i + 1) * 1.0e-6 + 0.25
        for _ in range(SERIAL_TRANSCENDENTAL_PAIRS):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0e-12)
        total += x

    return total


# ---------------------------------------------------------------------------
# Stage B: parallel worker; must be top-level for macOS 'spawn'
# ---------------------------------------------------------------------------
def parallel_worker(bounds):
    """Compute one contiguous chunk with 35 multiply-add/trig iterations."""
    start, end = bounds
    total = 0.0

    for i in range(start, end):
        x = (i % 1000 + 1) * 1.0e-4
        for k in range(PARALLEL_ITERATIONS):
            # One multiply-add style update and one transcendental operation.
            x = x * 1.0000001 + math.sin(x + k * 1.0e-6) * 1.0e-4
        total += x

    return total


def make_even_chunks(n, p):
    """Return p contiguous, evenly sized (or near-even) non-overlapping chunks."""
    base, remainder = divmod(n, p)
    chunks = []
    start = 0

    for worker_id in range(p):
        chunk_size = base + (1 if worker_id < remainder else 0)
        end = start + chunk_size
        chunks.append((start, end))
        start = end

    return chunks


# ---------------------------------------------------------------------------
# One complete pipeline run; the Pool itself is created outside this function
# and remains alive for all warm-up and timed repetitions at a given p.
# ---------------------------------------------------------------------------
def run_pipeline(pool, n, p, stage_a_samples):
    stage_start = time.perf_counter_ns()
    serial_result = serial_stage(n)
    stage_end = time.perf_counter_ns()

    stage_a_samples.append((stage_end - stage_start) / 1.0e6)

    chunks = make_even_chunks(n, p)
    partial_results = pool.map(parallel_worker, chunks, chunksize=1)

    # Consume the deterministic results so the complete computation is used.
    checksum = serial_result + sum(partial_results)
    if not math.isfinite(checksum):
        raise ArithmeticError("The pipeline produced a non-finite result")


def measure_pipeline(ctx, p, n, label):
    """Create a persistent pool, then benchmark the pipeline with it."""
    print(f"Measuring {label}: p={p}, N={n:,}", flush=True)
    stage_a_samples = []

    # Pool creation/startup is outside run_benchmark's timed region.
    with ctx.Pool(processes=p) as pool:
        stats = run_benchmark(
            lambda: run_pipeline(pool, n, p, stage_a_samples),
            warmup_runs=WARMUP_RUNS,
            timed_runs=TIMED_RUNS,
        )

    # run_benchmark makes 2 warm-ups followed by 7 timed invocations.
    timed_stage_samples = stage_a_samples[-TIMED_RUNS:]
    stage_a_median_ms = statistics.median(timed_stage_samples)

    return stats, stage_a_median_ms


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_strong_table(results):
    t1 = results[1][0].median_ms

    print("\nTABLE 2A: STRONG SCALING")
    print("-" * 82)
    print(
        f"{'p':<5}{'N':<14}{'Median (ms)':<16}"
        f"{'IQR (ms)':<13}{'Speedup':<13}{'Efficiency':<13}"
    )

    for p in sorted(results):
        stats, _stage_a = results[p]
        speedup = t1 / stats.median_ms
        efficiency = speedup / p
        print(
            f"{p:<5}{N_STRONG:<14,}{stats.median_ms:<16.4f}"
            f"{stats.iqr_ms:<13.4f}{speedup:<13.4f}{efficiency:<13.4f}"
        )


def print_weak_table(results):
    t1_weak = results[1][0].median_ms

    print("\nTABLE 2B: WEAK SCALING")
    print("-" * 90)
    print(
        f"{'p':<5}{'N(p)':<14}{'Median (ms)':<16}{'IQR (ms)':<13}"
        f"{'Efficiency':<14}{'Alpha':<12}{'Gustafson S':<14}"
    )

    for p in sorted(results):
        stats, stage_a_ms = results[p]
        n = p * N0_WEAK
        efficiency = t1_weak / stats.median_ms
        alpha = stage_a_ms / stats.median_ms
        gustafson = p - alpha * (p - 1)

        print(
            f"{p:<5}{n:<14,}{stats.median_ms:<16.4f}"
            f"{stats.iqr_ms:<13.4f}{efficiency:<14.4f}"
            f"{alpha:<12.6f}{gustafson:<14.4f}"
        )


def print_marginal_speedup(results):
    t1 = results[1][0].median_ms
    previous_gain = None

    print("\nMARGINAL SPEEDUP PER ADDED WORKER")
    print("-" * 66)
    print(f"{'Transition':<18}{'Added speedup':<20}{'Change vs prior gain (%)':<26}")

    for p in range(2, P_LOG + 1):
        previous_speedup = t1 / results[p - 1][0].median_ms
        current_speedup = t1 / results[p][0].median_ms
        current_gain = current_speedup - previous_speedup

        if previous_gain is None or abs(previous_gain) < 1.0e-12:
            change_text = "N/A"
        else:
            change = (current_gain - previous_gain) / abs(previous_gain) * 100.0
            change_text = f"{change:.2f}"

        print(f"{p - 1} -> {p:<13}{current_gain:<20.4f}{change_text:<26}")
        previous_gain = current_gain


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    if P_LOG < 1:
        raise RuntimeError(f"Invalid logical CPU count: {P_LOG}")

    ctx = mp.get_context("spawn")

    print("=" * 70)
    print("PHASE 2: STRONG SCALING AND WEAK SCALING")
    print("=" * 70)
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPUs: {P_LOG}")
    print(f"Strong scaling N: {N_STRONG:,}")
    print(f"Weak scaling N0 per process: {N0_WEAK:,}")
    print(f"Serial prefix: {SERIAL_PREFIX_FRACTION:.0%}")
    print(f"Parallel iterations per element: {PARALLEL_ITERATIONS}")
    print(f"Benchmark repetitions: {WARMUP_RUNS} warm-up + {TIMED_RUNS} timed")

    # Experiment 2A: fixed N, every integer process count from 1 to P_LOG.
    strong_results = {}
    print("\nStarting Experiment 2A: Strong Scaling", flush=True)
    for p in range(1, P_LOG + 1):
        strong_results[p] = measure_pipeline(
            ctx, p, N_STRONG, "strong scaling"
        )

    print_strong_table(strong_results)

    # Experiment 2B: fixed work per worker; total N grows with p.
    weak_results = {}
    print("\nStarting Experiment 2B: Weak Scaling", flush=True)
    for p in range(1, P_LOG + 1):
        n = p * N0_WEAK
        weak_results[p] = measure_pipeline(
            ctx, p, n, "weak scaling"
        )

    print_weak_table(weak_results)

    # Amdahl analysis from the p=1 strong-scaling run.
    t1_strong = strong_results[1][0].median_ms
    t_stage_a_1 = strong_results[1][1]
    serial_fraction = t_stage_a_1 / t1_strong

    if not 0.0 < serial_fraction <= 1.0:
        raise ArithmeticError(
            f"Unexpected measured serial fraction: {serial_fraction}"
        )

    amdahl_max = 1.0 / serial_fraction
    amdahl_at_plog = 1.0 / (
        serial_fraction + (1.0 - serial_fraction) / P_LOG
    )
    observed_at_plog = t1_strong / strong_results[P_LOG][0].median_ms

    print("\nAMDAHL'S LAW ANALYSIS")
    print("-" * 55)
    print(f"Stage A median at p=1:       {t_stage_a_1:.4f} ms")
    print(f"Total median at p=1:         {t1_strong:.4f} ms")
    print(f"Measured serial fraction s:  {serial_fraction:.6f}")
    print(f"Theoretical maximum S(inf):  {amdahl_max:.4f}")
    print(f"Theoretical speedup at p={P_LOG}: {amdahl_at_plog:.4f}")
    print(f"Observed speedup at p={P_LOG}:   {observed_at_plog:.4f}")

    print_marginal_speedup(strong_results)

    # Gustafson scaled speedup at the largest worker count.
    weak_last_stats, weak_last_stage_a = weak_results[P_LOG]
    weak_last_alpha = weak_last_stage_a / weak_last_stats.median_ms
    gustafson_at_plog = P_LOG - weak_last_alpha * (P_LOG - 1)

    print("\nGUSTAFSON ANALYSIS")
    print("-" * 55)
    print(f"Stage A median at p={P_LOG}: {weak_last_stage_a:.4f} ms")
    print(f"Total weak median at p={P_LOG}: {weak_last_stats.median_ms:.4f} ms")
    print(f"Alpha at p={P_LOG}: {weak_last_alpha:.6f}")
    print(f"Gustafson scaled speedup: {gustafson_at_plog:.4f}")

    print("\nPHASE 2 VERIFICATION")
    print("-" * 55)
    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            observed_at_plog,
            gustafson_at_plog,
        )
    )
    print("\nPHASE 2 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
