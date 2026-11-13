"""Phase 2: strong (Amdahl) vs weak (Gustafson) scaling of a serial-A / parallel-B pipeline."""
import math, time, multiprocessing as mp
from harness import emit_verify_tag
from common import STUDENT_ID, QUICK, hw_info, save

N_STRONG = 120_000 if QUICK else 1_200_000
N0 = 20_000 if QUICK else 200_000
WARMUP, RUNS = (0, 3) if QUICK else (1, 5)

def stage_a(n):
    """Serial: 10 math.sin/math.sqrt ops per element."""
    acc = 0.0
    sin, sqrt = math.sin, math.sqrt
    for i in range(n):
        x = i * 1e-6 + 1.0
        x = sin(x); x = sqrt(x * x + 1.0); x = sin(x); x = sqrt(x * x + 1.0); x = sin(x)
        x = sqrt(x * x + 1.0); x = sin(x); x = sqrt(x * x + 1.0); x = sin(x); x = sqrt(x * x + 1.0)
        acc += x
    return acc

def stage_b(rng):
    """Parallel chunk: 30 FMA/trig ops per element, no shared writes."""
    lo, hi = rng
    sin, cos = math.sin, math.cos
    acc = 0.0
    for i in range(lo, hi):
        x = i * 1e-6
        y = x * 1.000001 + 0.5; y = y * 0.999999 + x; y = y * 1.000002 + 0.25; y = y * 0.999998 + x
        y = y * 1.000003 + 0.125; y = y * 0.999997 + x; y = y * 1.000004 + 0.0625; y = y * 0.999996 + x
        y = y * 1.000005 + 0.03; y = y * 0.999995 + x
        z = sin(y); z = cos(z); z = sin(z); z = cos(z); z = sin(z)
        z = cos(z); z = sin(z); z = cos(z); z = sin(z); z = cos(z)
        w = z * y + x; w = w * 0.5 + z; w = w * 0.5 + y; w = w * 0.5 + z; w = w * 0.5 + y
        w = w * 0.5 + z; w = w * 0.5 + y; w = w * 0.5 + z; w = w * 0.5 + y; w = w * 0.5 + z
        acc += w
    return acc

def chunks(n, p):
    b = [n * t // p for t in range(p + 1)]
    return [(b[t], b[t + 1]) for t in range(p)]

def trial_series(pool, n, p):
    """Returns median T_total(ms), median T_A(ms)."""
    tot, ta = [], []
    for k in range(WARMUP + RUNS):
        t0 = time.perf_counter_ns()
        stage_a(int(0.05 * n))
        t1 = time.perf_counter_ns()
        pool.map(stage_b, chunks(n, p), chunksize=1)
        t2 = time.perf_counter_ns()
        if k >= WARMUP:
            tot.append((t2 - t0) / 1e6); ta.append((t1 - t0) / 1e6)
    tot.sort(); ta.sort()
    return tot[RUNS // 2], ta[RUNS // 2], tot[(3 * RUNS) // 4] - tot[RUNS // 4]

def main():
    hw = hw_info(); P_phys, P_log = hw["P_phys"], mp.cpu_count()
    strong, weak = [], []
    for p in range(1, P_log + 1):
        with mp.Pool(processes=p) as pool:          # created outside the timed loop
            pool.map(stage_b, chunks(1000 * p, p))  # make sure workers are up
            T, TA, iqr = trial_series(pool, N_STRONG, p)
            strong.append({"p": p, "T_ms": T, "TA_ms": TA, "iqr_ms": iqr})
            Tw, TAw, iqrw = trial_series(pool, p * N0, p)
            weak.append({"p": p, "N": p * N0, "T_ms": Tw, "TA_ms": TAw, "iqr_ms": iqrw})
        print(f"p={p:2d}  strong T={T:8.1f} ms (A={TA:6.1f})   weak T={Tw:8.1f} ms (A={TAw:6.1f})")
    T1, Tw1 = strong[0]["T_ms"], weak[0]["T_ms"]
    s = strong[0]["TA_ms"] / T1
    for r in strong:
        r["S"] = T1 / r["T_ms"]; r["E"] = r["S"] / r["p"]
    for r in weak:
        r["Ew"] = Tw1 / r["T_ms"]; r["alpha"] = r["TA_ms"] / r["T_ms"]
        r["SG"] = r["p"] - r["alpha"] * (r["p"] - 1)
    amdahl = 1 / s
    print(f"serial fraction s={s:.4f}  Amdahl cap={amdahl:.2f}  S_strong(P_log)={strong[-1]['S']:.2f}  "
          f"S_Gustafson(P_log)={weak[-1]['SG']:.2f}")
    # Inflection: SMT boundary (P_phys -> P_phys+1) or, on hybrid CPUs, P-core -> E-core boundary
    infl = None
    b = P_phys if P_log > P_phys else (hw.get("P_cores") if hw.get("P_cores") and hw["P_cores"] < P_log else None)
    if b:
        S = {r["p"]: r["S"] for r in strong}
        infl = {"boundary": b, "kind": "SMT" if P_log > P_phys else "P-core -> E-core",
                "dS_before": S[b] - S.get(b - 1, 0.0), "dS_after": S[b + 1] - S[b]}
        print(f"{infl['kind']} boundary p={b}: dS {b-1}->{b}: {infl['dS_before']:+.3f} | {b}->{b+1}: {infl['dS_after']:+.3f}")
    tag = emit_verify_tag("PHASE2", STUDENT_ID, strong[-1]["S"], weak[-1]["SG"])
    print(tag)
    save("phase2", {"N": N_STRONG, "N0": N0, "s": s, "amdahl_cap": amdahl, "strong": strong,
                    "weak": weak, "inflection": infl, "P_phys": P_phys, "tag": tag})

if __name__ == "__main__":
    main()
