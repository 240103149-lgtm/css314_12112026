
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag


# ==========================================
# HARDWARE AND EXPERIMENT CONFIGURATION
# ==========================================

STUDENT_ID = 240103149

CPU_NAME = "Apple M3"
P_PHYS = 8
P_LOG = mp.cpu_count()

CACHE_LINE_BYTES = 128

TOTAL_OPS = 24_000_000

N_SLOTS = 512
SHM_SIZE = 4096

STRIDES = (1, 2, 4, 8, 16, 32)


# ==========================================
# WORKER FUNCTION
# ==========================================

def fs_worker(shm_name, slot_idx, iters):

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = None

    try:
        arr = (ctypes.c_uint64 * N_SLOTS).from_buffer(shm.buf)

        acc = arr[slot_idx]

        mask = 0xFFFFFFFFFFFFFFFF
        mult = 6364136223846793005

        for i in range(iters):

            acc = (acc * mult + (i | 1)) & mask

            arr[slot_idx] = acc

    finally:

        # Release exported buffer before closing shared memory
        if arr is not None:
            del arr

        shm.close()


# ==========================================
# SINGLE BENCHMARK TRIAL
# ==========================================

def run_trial(p, stride):

    if TOTAL_OPS % p != 0:
        raise ValueError("TOTAL_OPS must be divisible by p")

    if (p - 1) * stride >= N_SLOTS:
        raise ValueError("Shared memory slot index is out of range")

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    processes = []

    try:

        # Initialize the worker slots safely.
        # Avoid memoryview slice assignment.
        slots = (ctypes.c_uint64 * N_SLOTS).from_buffer(shm.buf)

        try:
            for t in range(p):
                slots[t * stride] = 0
        finally:
            del slots

        iters = TOTAL_OPS // p

        ctx = mp.get_context("spawn")

        # Start workers
        for t in range(p):

            slot_idx = t * stride

            proc = ctx.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            proc.start()
            processes.append(proc)

        # Wait for workers
        for proc in processes:
            proc.join()

        # Check worker exit codes
        failed = [
            (proc.pid, proc.exitcode)
            for proc in processes
            if proc.exitcode != 0
        ]

        if failed:
            raise RuntimeError(
                f"Worker process failed: {failed}"
            )

    finally:

        # Stop any remaining processes
        for proc in processes:

            if proc.is_alive():
                proc.terminate()
                proc.join()

        # Release shared memory
        shm.close()
        shm.unlink()


# ==========================================
# BENCHMARK MEASUREMENT
# ==========================================

def measure(p, stride):

    print(
        f"Measuring: processes={p}, stride={stride}",
        flush=True
    )

    stats = run_benchmark(

        lambda: run_trial(p, stride),

        warmup_runs=2,

        timed_runs=7
    )

    return stats


# ==========================================
# HELPER: PROCESS COUNTS
# ==========================================

def get_process_counts():

    counts = []

    p = 1

    while p <= P_LOG:
        counts.append(p)
        p *= 2

    if counts[-1] != P_LOG:
        counts.append(P_LOG)

    return counts


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    print("=" * 65)
    print("PHASE 1: FALSE SHARING EXPERIMENT")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"CPU: {CPU_NAME}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical CPUs: {P_LOG}")
    print(f"Cache line size: {CACHE_LINE_BYTES} bytes")
    print(f"Total operations: {TOTAL_OPS:,}")

    # ======================================
    # TABLE 1A: STRIDE SENSITIVITY
    # ======================================

    print("\nTABLE 1A: STRIDE SENSITIVITY")

    print("-" * 75)

    print(
        f"{'Stride':<10}"
        f"{'Offset (bytes)':<20}"
        f"{'Median (ms)':<16}"
        f"{'IQR (ms)':<13}"
        f"{'Mops/s':<12}"
    )

    stride_results = {}

    for stride in STRIDES:

        stats = measure(P_PHYS, stride)

        stride_results[stride] = stats

        offset = stride * 8

        mops = (
            (TOTAL_OPS / 1_000_000)
            / (stats.median_ms / 1000)
        )

        print(
            f"{stride:<10}"
            f"{offset:<20}"
            f"{stats.median_ms:<16.4f}"
            f"{stats.iqr_ms:<13.4f}"
            f"{mops:<12.2f}"
        )

    # ======================================
    # TABLE 1B: KAR P-FLATT SCALING
    # ======================================

    process_counts = get_process_counts()

    scaling_results = {}

    print("\nTABLE 1B: PROCESS SCALING")

    for stride in (1, 16):

        print("\n" + "=" * 65)
        print(f"STRIDE = {stride}")
        print("=" * 65)

        results = {}

        for p in process_counts:

            # Reuse the measurement at P_PHYS
            # already obtained in Table 1A.
            if p == P_PHYS:

                stats = stride_results[stride]

            else:

                stats = measure(p, stride)

            results[p] = stats

        scaling_results[stride] = results

        t1 = results[1].median_ms

        print(
            f"\n{'p':<8}"
            f"{'Median (ms)':<16}"
            f"{'Speedup':<14}"
            f"{'Efficiency':<14}"
            f"{'Karp-Flatt e(p)':<16}"
        )

        for p in process_counts:

            tp = results[p].median_ms

            speedup = t1 / tp

            efficiency = speedup / p

            if p == 1:

                karp_flatt = float("nan")
                kf_text = "N/A"

            else:

                karp_flatt = (
                    (1 / speedup - 1 / p)
                    / (1 - 1 / p)
                )

                kf_text = f"{karp_flatt:.6f}"

            print(
                f"{p:<8}"
                f"{tp:<16.4f}"
                f"{speedup:<14.4f}"
                f"{efficiency:<14.4f}"
                f"{kf_text:<16}"
            )

    # ======================================
    # VERIFY HASH
    # ======================================

    median_stride1 = stride_results[1].median_ms

    median_stride16 = stride_results[16].median_ms

    print("\n" + "=" * 65)
    print("PHASE 1 VERIFICATION")
    print("=" * 65)

    verify = emit_verify_tag(

        "PHASE1",

        STUDENT_ID,

        median_stride1,

        median_stride16
    )

    print(verify)

    print("\nPHASE 1 COMPLETED")


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    mp.freeze_support()

    main()

