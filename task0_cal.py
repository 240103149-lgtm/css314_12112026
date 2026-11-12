
import multiprocessing as mp
import time

from harness import emit_verify_tag


def main():
    student_id = 240103149
    reads = 2_000_000

    logical_cpus = mp.cpu_count()

    start = time.perf_counter_ns()

    for _ in range(reads):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_latency_ns = total_ns / reads
    total_ms = total_ns / 1_000_000

    print("PHASE 0 RESULTS")
    print("--------------------------")
    print(f"Student ID: {student_id}")
    print(f"Logical CPUs: {logical_cpus}")
    print(f"Clock reads: {reads}")
    print(f"Total time: {total_ms:.4f} ms")
    print(f"Average call latency: {average_latency_ns:.4f} ns")

    verify = emit_verify_tag(
        "PHASE0",
        student_id,
        average_latency_ns,
        total_ms
    )

    print(verify)


if __name__ == "__main__":
    main()

