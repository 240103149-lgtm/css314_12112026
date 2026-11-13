"""Phase 3: empirical roofline (BW + FLOP ceilings) and progressive GEMM with tile sweep."""
import random, time, multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag
from common import STUDENT_ID, QUICK, hw_info, save

BUF = 64 * 10**6                      # 64 MB per buffer (128 MB total)
FLOPS_PER_WORKER = 6_000_000 if QUICK else 60_000_000
N = 96 if QUICK else 256
BS_LIST = [8, 16, 32, 64, 96, 128]
WARMUP, RUNS = (0, 3) if QUICK else (1, 5)

_G = {}

# ---------- 3A: roofline ----------
def _init_bw(na, nb):
    _G["a"] = shared_memory.SharedMemory(name=na); _G["b"] = shared_memory.SharedMemory(name=nb)
    _G["mva"] = _G["a"].buf; _G["mvb"] = _G["b"].buf

def bw_copy(rng):
    s, e = rng
    _G["mva"][s:e] = _G["mvb"][s:e]
    return e - s

def fma_kernel(flops):
    a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = 0.0
    a8 = a9 = a10 = a11 = a12 = a13 = a14 = a15 = 0.0
    m, c = 0.999999, 1e-7
    for _ in range(flops // 32):          # 16 FMAs x 2 FLOPs = 32 FLOPs / iteration
        a0 = a0 * m + c; a1 = a1 * m + c; a2 = a2 * m + c; a3 = a3 * m + c
        a4 = a4 * m + c; a5 = a5 * m + c; a6 = a6 * m + c; a7 = a7 * m + c
        a8 = a8 * m + c; a9 = a9 * m + c; a10 = a10 * m + c; a11 = a11 * m + c
        a12 = a12 * m + c; a13 = a13 * m + c; a14 = a14 * m + c; a15 = a15 * m + c
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15

# ---------- 3B: GEMM ----------
def _init_gemm(na, nb, nc, n):
    for k, nm in (("A", na), ("B", nb), ("C", nc)):
        _G["shm" + k] = shared_memory.SharedMemory(name=nm)
        _G[k] = _G["shm" + k].buf[:n * n * 8].cast("d")
    _G["N"] = n

def gemm_rows(job):
    variant, lo, hi, BS = job
    A, B, C, n = _G["A"], _G["B"], _G["C"], _G["N"]
    if variant == 1:                                   # naive i-j-k, strided B[k*N+j]
        for i in range(lo, hi):
            iN = i * n
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[iN + k] * B[k * n + j]
                C[iN + j] = s
        return
    for idx in range(lo * n, hi * n):
        C[idx] = 0.0
    if variant == 2:                                   # reordered i-k-j, unit-stride inner j
        for i in range(lo, hi):
            iN = i * n
            for k in range(n):
                a_ik = A[iN + k]; kN = k * n
                for j in range(n):
                    C[iN + j] += a_ik * B[kN + j]
        return
    # variant 3: 6-loop cache-blocked, (i,k,j) micro-kernel, a_ik hoisted
    for ii in range(lo, hi, BS):
        i_end = min(ii + BS, hi)
        for kk in range(0, n, BS):
            k_end = min(kk + BS, n)
            for jj in range(0, n, BS):
                j_end = min(jj + BS, n)
                for i in range(ii, i_end):
                    iN = i * n
                    for k in range(kk, k_end):
                        a_ik = A[iN + k]; kN = k * n
                        for j in range(jj, j_end):
                            C[iN + j] += a_ik * B[kN + j]

def main():
    hw = hw_info(); P = hw["P_phys"]
    out = {"P_phys": P, "N": N}
    # ---- 3A DRAM bandwidth ----
    sa = shared_memory.SharedMemory(create=True, size=BUF); sb = shared_memory.SharedMemory(create=True, size=BUF)
    try:
        sb.buf[:BUF] = bytes(range(256)) * (BUF // 256)   # macOS rounds shm size up to page size
        rngs = [(BUF * t // P, BUF * (t + 1) // P) for t in range(P)]
        with mp.Pool(P, initializer=_init_bw, initargs=(sa.name, sb.name)) as pool:
            st = run_benchmark(lambda: pool.map(bw_copy, rngs, chunksize=1), WARMUP + 1, RUNS)
    finally:
        for s in (sa, sb): s.close(); s.unlink()
    bw = 2 * BUF / (st.median_ms / 1e3 * 1e9)
    # ---- 3A compute ceiling ----
    with mp.Pool(P) as pool:
        stc = run_benchmark(lambda: pool.map(fma_kernel, [FLOPS_PER_WORKER] * P, chunksize=1), WARMUP, RUNS)
    peak = P * FLOPS_PER_WORKER / (stc.median_ms / 1e3 * 1e9)
    ridge = peak / bw
    out.update(bw_gbs=bw, bw_ms=st.median_ms, peak_gflops=peak, fma_ms=stc.median_ms, ridge=ridge)
    print(f"BW_sustained={bw:.2f} GB/s  P_sustained={peak:.4f} GFLOP/s  ridge I*={ridge:.5f} FLOP/B")

    # ---- 3B GEMM ----
    rnd = random.Random(STUDENT_ID or 1)
    nbytes = N * N * 8
    shm = {k: shared_memory.SharedMemory(create=True, size=nbytes) for k in "ABC"}
    try:
        A, B = shm["A"].buf[:nbytes].cast("d"), shm["B"].buf[:nbytes].cast("d")
        for i in range(N * N):
            A[i] = rnd.uniform(-1, 1); B[i] = rnd.uniform(-1, 1)
        A.release(); B.release()
        rows = [(N * t // P, N * (t + 1) // P) for t in range(P)]
        flop = 2 * N ** 3
        gemm = []
        with mp.Pool(P, initializer=_init_gemm, initargs=(shm["A"].name, shm["B"].name, shm["C"].name, N)) as pool:
            def run(variant, BS=0):
                jobs = [(variant, lo, hi, BS) for lo, hi in rows]
                return run_benchmark(lambda: pool.map(gemm_rows, jobs, chunksize=1), WARMUP, RUNS)
            C = shm["C"].buf[:nbytes].cast("d")
            for v in (1, 2):
                s = run(v)
                gemm.append({"variant": v, "BS": None, "median_ms": s.median_ms, "iqr_ms": s.iqr_ms,
                             "gflops": flop / (s.median_ms / 1e3 * 1e9)})
                print(f"Variant {v}: {s.median_ms:9.1f} ms  {gemm[-1]['gflops']:.4f} GFLOP/s")
            C2 = list(C)                                    # reference from Variant 2
            for bs in BS_LIST:
                s = run(3, bs)
                err = max(abs(x - y) for x, y in zip(C, C2))
                gemm.append({"variant": 3, "BS": bs, "median_ms": s.median_ms, "iqr_ms": s.iqr_ms,
                             "gflops": flop / (s.median_ms / 1e3 * 1e9), "max_err": err,
                             "W_tile_B": 3 * bs * bs * 8, "I_tiled": bs / 12})
                assert err < 1e-9, f"Variant 3 mismatch at BS={bs}: {err}"
                print(f"Variant 3 BS={bs:3d}: {s.median_ms:9.1f} ms  {gemm[-1]['gflops']:.4f} GFLOP/s  "
                      f"W_tile={3*bs*bs*8/1024:.1f} KiB  max|dC|={err:.1e}")
            C.release()
    finally:
        for s in shm.values(): s.close(); s.unlink()
    v3 = [g for g in gemm if g["variant"] == 3]
    best = max(v3, key=lambda g: g["gflops"])
    out.update(gemm=gemm, BS_star=best["BS"])
    g1 = gemm[0]["gflops"]
    tag = emit_verify_tag("PHASE3", STUDENT_ID, g1, best["gflops"])
    print(f"BS*={best['BS']}  W_tile={best['W_tile_B']/1024:.1f} KiB  I_tiled~{best['I_tiled']:.2f} FLOP/B")
    print(tag)
    out["tag"] = tag
    save("phase3", out)

if __name__ == "__main__":
    main()
