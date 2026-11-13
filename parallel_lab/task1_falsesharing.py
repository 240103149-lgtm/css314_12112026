"""Phase 1: false-sharing stride sweep + process scaling with Karp-Flatt."""
import ctypes, multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag
from common import STUDENT_ID, QUICK, hw_info, save

M = 1_600_000 if QUICK else 16_000_000
WARMUP, RUNS = (0, 3) if QUICK else (2, 7)
STRIDES = [1, 2, 4, 8, 16, 32]

def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc  # Forces store to shared L1 cache line
    del arr
    shm.close()

def make_trial(shm_name, p, stride):
    iters = M // p
    def trial():
        procs = [mp.Process(target=fs_worker, args=(shm_name, (t * stride) % 512, iters)) for t in range(p)]
        for pr in procs: pr.start()
        for pr in procs: pr.join()
    return trial

def karp_flatt(S, p):
    return (1 / S - 1 / p) / (1 - 1 / p) if p >= 2 else None

def main():
    hw = hw_info(); P_phys, P_log = hw["P_phys"], mp.cpu_count()
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        # 1. Stride sweep at p = P_phys
        sweep = []
        for s in STRIDES:
            st = run_benchmark(make_trial(shm.name, P_phys, s), WARMUP, RUNS)
            row = {"stride": s, "bytes": s * 8, "median_ms": st.median_ms, "iqr_ms": st.iqr_ms,
                   "min_ms": st.min_ms, "mops": M / (st.median_ms / 1e3) / 1e6}
            sweep.append(row)
            print(f"stride={s:2d} ({s*8:3d} B): median={st.median_ms:9.1f} ms  IQR={st.iqr_ms:7.1f}  {row['mops']:.2f} Mops/s")
        # 2. Process scaling p in {1,2,4,...,P_log} (+P_log itself if not a power of 2)
        ps = []; p = 1
        while p <= P_log: ps.append(p); p *= 2
        if ps[-1] != P_log: ps.append(P_log)
        scaling = {}
        for s in (1, 16):
            rows = []
            for p in ps:
                st = run_benchmark(make_trial(shm.name, p, s), WARMUP, RUNS)
                rows.append({"p": p, "T_ms": st.median_ms, "iqr_ms": st.iqr_ms})
            T1 = rows[0]["T_ms"]
            for r in rows:
                r["S"] = T1 / r["T_ms"]; r["E"] = r["S"] / r["p"]; r["e"] = karp_flatt(r["S"], r["p"])
                e = f"{r['e']:.3f}" if r["e"] is not None else "  -  "
                print(f"stride={s:2d} p={r['p']:2d}: T={r['T_ms']:9.1f} ms S={r['S']:.2f} E={r['E']:.2f} e={e}")
            scaling[str(s)] = rows
    finally:
        shm.close(); shm.unlink()
    m1 = next(r["median_ms"] for r in sweep if r["stride"] == 1)
    m2 = next(r["median_ms"] for r in sweep if r["stride"] == 16)
    tag = emit_verify_tag("PHASE1", STUDENT_ID, m1, m2)
    print(tag)
    save("phase1", {"M": M, "P_phys": P_phys, "sweep": sweep, "scaling": scaling, "tag": tag})

if __name__ == "__main__":
    main()
