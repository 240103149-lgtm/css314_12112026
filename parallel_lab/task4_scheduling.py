"""Phase 4: triangular skewed workload - block static vs cyclic static vs dynamic chunk queue."""
import math, time, multiprocessing as mp
from harness import emit_verify_tag
from common import STUDENT_ID, QUICK, hw_info, save

N = 2_000 if QUICK else 8_000
CHUNKS = [1, 8, 64, 512]
WARMUP, RUNS = (0, 1) if QUICK else (0, 3)

def K(i):
    return 1 + int((i / 40) ** 2)

def work(i):
    x = i * 1e-3
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(K(i)):
        x = sqrt(abs(sin(x) * cos(x)) + 1.0)
    return x

def w_block(t, p, times, *_):
    t0 = time.perf_counter()
    for i in range(N * t // p, N * (t + 1) // p): work(i)
    times[t] = time.perf_counter() - t0

def w_cyclic(t, p, times, *_):
    t0 = time.perf_counter()
    for i in range(t, N, p): work(i)
    times[t] = time.perf_counter() - t0

def w_dynamic(t, p, times, counter, C):
    t0 = time.perf_counter()
    while True:
        with counter.get_lock():
            s = counter.value; counter.value = s + C
        if s >= N: break
        for i in range(s, min(s + C, N)): work(i)
    times[t] = time.perf_counter() - t0

def run(target, p, C=0):
    walls, lams, tws = [], [], []
    for k in range(WARMUP + RUNS):
        times = mp.Array("d", p); counter = mp.Value("i", 0)
        procs = [mp.Process(target=target, args=(t, p, times, counter, C)) for t in range(p)]
        t0 = time.perf_counter()
        for pr in procs: pr.start()
        for pr in procs: pr.join()
        wall = time.perf_counter() - t0
        if k >= WARMUP:
            tw = list(times); walls.append(wall); tws.append(tw)
            lams.append(max(tw) / (sum(tw) / p))
    i = sorted(range(RUNS), key=lambda j: walls[j])[RUNS // 2]   # median run
    return {"T_wall_s": walls[i], "lambda": lams[i], "t_worker": tws[i]}

def main():
    hw = hw_info(); p = hw["P_phys"]
    rows = []
    for name, tgt, C in [("Block static", w_block, None), ("Cyclic static", w_cyclic, None)] + \
                        [("Dynamic", w_dynamic, c) for c in CHUNKS]:
        r = run(tgt, p, C or 0); r.update(strategy=name, C=C); rows.append(r)
        lab = f"{name}" + (f" C={C}" if C else "")
        print(f"{lab:20s} T_wall={r['T_wall_s']:7.3f} s  lambda={r['lambda']:.3f}  "
              f"t_worker=[{', '.join(f'{x:.2f}' for x in r['t_worker'])}]")
    dyn = [r for r in rows if r["strategy"] == "Dynamic"]
    best = min(dyn, key=lambda r: r["T_wall_s"])
    tag = emit_verify_tag("PHASE4", STUDENT_ID, rows[0]["T_wall_s"], best["T_wall_s"])
    print(f"C* = {best['C']}")
    print(tag)
    save("phase4", {"N": N, "p": p, "rows": rows, "C_star": best["C"],
                    "total_iters": sum(K(i) for i in range(N)), "tag": tag})

if __name__ == "__main__":
    main()
