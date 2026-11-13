"""Phase 0: hardware topology + perf_counter_ns() call-latency calibration."""
import multiprocessing as mp, time
from harness import emit_verify_tag
from common import STUDENT_ID, hw_info, save

def main():
    hw = hw_info()
    p_log = mp.cpu_count()
    n = 1_000_000
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(n):
        pc()
    avg_ns = (pc() - t0) / n
    tag = emit_verify_tag("PHASE0", STUDENT_ID, avg_ns, float(p_log))
    print(f"CPU: {hw['model']} | P_phys={hw['P_phys']} P_log={p_log} | line={hw['line']} B")
    print(f"L1d={hw['L1d']} L2={hw['L2']} L3={hw['L3']}")
    print(f"perf_counter_ns avg latency: {avg_ns:.2f} ns/call")
    print(tag)
    save("phase0", {"avg_ns": avg_ns, "P_log": p_log, "tag": tag})

if __name__ == "__main__":
    main()
