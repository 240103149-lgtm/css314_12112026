"""Builds lab_report_<ID>.md from results/*.json + source files."""
from pathlib import Path
from common import STUDENT_ID, hw_info, load

def kib(b): return "n/a" if not b else (f"{b/1024**2:.0f} MiB" if b >= 1024**2 else f"{b/1024:g} KiB")
def f(x, d=2): return "-" if x is None else f"{x:.{d}f}"

def main():
    hw = hw_info(); p0, p1, p2, p3, p4 = (load(f"phase{i}") for i in range(5))
    L = []; w = L.append
    w(f"# Lab Report — Performance Evaluation and Optimization in Parallel Computing\n")
    w(f"**Student ID:** {STUDENT_ID}  \n**OS / Python:** {hw['os']} / {hw['python']}\n")
    w("## 1. Phase 0 — Hardware & verification tags\n")
    w("| CPU Model | P_phys | P_log | Cache line | L1d | L2 | L3 |\n|---|---|---|---|---|---|---|")
    w(f"| {hw['model']} | {hw['P_phys']} | {hw['P_log']} | {hw['line']} B | {kib(hw['L1d'])} | {kib(hw['L2'])} | {kib(hw['L3'])} |\n")
    if hw.get("P_cores") or hw.get("E_cores"):
        w(f"Hybrid topology: P-cores={hw.get('P_cores')}, E-cores={hw.get('E_cores')}\n")
    w(f"`perf_counter_ns()` average call latency (10^6 calls): **{p0['avg_ns']:.2f} ns**\n")
    w("```")
    for p in (p0, p1, p2, p3, p4):
        if p: w(p["tag"])
    w("```\n")
    w("## 2. Measured results\n")
    # ---- Phase 1
    if p1:
        w(f"### Phase 1 — False sharing (M = {p1['M']:.1e}, p = P_phys = {p1['P_phys']})\n")
        w("| stride | offset (B) | median ms | IQR ms | Mops/s |\n|---|---|---|---|---|")
        for r in p1["sweep"]:
            w(f"| {r['stride']} | {r['bytes']} | {r['median_ms']:.1f} | {r['iqr_ms']:.1f} | {r['mops']:.2f} |")
        w("")
        w("| p | T(s=1) ms | S | E | e(p) | T(s=16) ms | S | E | e(p) |\n|---|---|---|---|---|---|---|---|---|")
        for a, b in zip(p1["scaling"]["1"], p1["scaling"]["16"]):
            w(f"| {a['p']} | {a['T_ms']:.1f} | {a['S']:.2f} | {a['E']:.2f} | {f(a['e'],3)} | "
              f"{b['T_ms']:.1f} | {b['S']:.2f} | {b['E']:.2f} | {f(b['e'],3)} |")
        w("")
    if p2:
        w(f"### Phase 2 — Strong (N = {p2['N']:.1e}) and weak (N0 = {p2['N0']:.1e}) scaling\n")
        w(f"Serial fraction s = T_A(1)/T_1 = **{p2['s']:.4f}** → Amdahl cap S∞ = **{p2['amdahl_cap']:.2f}**\n")
        w("| p | T_strong ms | S_strong | E_strong | T_weak ms | E_weak | α | S_Gustafson |\n|---|---|---|---|---|---|---|---|")
        for a, b in zip(p2["strong"], p2["weak"]):
            w(f"| {a['p']} | {a['T_ms']:.1f} | {a['S']:.2f} | {a['E']:.2f} | {b['T_ms']:.1f} | {b['Ew']:.2f} | {b['alpha']:.4f} | {b['SG']:.2f} |")
        w("")
    if p3:
        w(f"### Phase 3 — Roofline & GEMM (N = {p3['N']}, p = {p3['P_phys']})\n")
        w("| Ceiling | Value |\n|---|---|")
        w(f"| BW_sustained | {p3['bw_gbs']:.2f} GB/s |\n| P_sustained (Python scalar FMA) | {p3['peak_gflops']:.4f} GFLOP/s |\n| Ridge I* | {p3['ridge']:.5f} FLOP/B |\n")
        w("| Variant | BS | median ms | GFLOP/s | W_tile | I_tiled ≈ BS/12 | max abs(C3−C2) |\n|---|---|---|---|---|---|---|")
        for g in p3["gemm"]:
            nm = {1: "1 naive i-j-k", 2: "2 reordered i-k-j", 3: "3 blocked"}[g["variant"]]
            w(f"| {nm} | {g['BS'] or '-'} | {g['median_ms']:.1f} | {g['gflops']:.4f} | "
              f"{kib(g.get('W_tile_B')) if g.get('W_tile_B') else '-'} | {f(g.get('I_tiled'))} | "
              f"{('%.1e' % g['max_err']) if 'max_err' in g else '-'} |")
        w("")
    if p4:
        w(f"### Phase 4 — Skewed workload (N = {p4['N']}, p = {p4['p']}, ΣK = {p4['total_iters']:,})\n")
        w("| Strategy | C | T_wall s | λ | t_worker s |\n|---|---|---|---|---|")
        for r in p4["rows"]:
            w(f"| {r['strategy']} | {r['C'] or '-'} | {r['T_wall_s']:.3f} | {r['lambda']:.3f} | "
              f"{', '.join(f'{x:.2f}' for x in r['t_worker'])} |")
        w("")
    # ---- Answers
    w("## 3. Diagnostic answers\n")
    if p1:
        sw = p1["sweep"]; line = hw["line"]
        inside = [r for r in sw if r["bytes"] < line]; outside = [r for r in sw if r["bytes"] >= line]
        mi = sum(r["mops"] for r in inside) / max(len(inside), 1); mo = sum(r["mops"] for r in outside) / max(len(outside), 1)
        jump = max(range(1, len(sw)), key=lambda k: sw[k]["mops"] / sw[k-1]["mops"])
        e1 = [r["e"] for r in p1["scaling"]["1"] if r["e"] is not None]
        e16 = [r["e"] for r in p1["scaling"]["16"] if r["e"] is not None]
        ratio = mo / mi
        if ratio > 1.05:
            w(f"**Phase 1.** Largest throughput step is at stride {sw[jump-1]['stride']}→{sw[jump]['stride']} "
              f"({sw[jump-1]['bytes']}→{sw[jump]['bytes']} B). Slots < {line} B apart share one cache line "
              f"(mean {mi:.2f} Mops/s); slots ≥ {line} B apart are on separate lines (mean {mo:.2f} Mops/s), "
              f"ratio ×{ratio:.2f}. Every store to a shared line invalidates the copy in the other cores (MESI ping-pong). ")
        else:
            w(f"**Phase 1.** No sharp drop is visible: slots < {line} B apart (same line) give {mi:.2f} Mops/s, slots ≥ {line} B "
              f"give {mo:.2f} Mops/s (ratio ×{ratio:.2f}, within noise). Each CPython iteration costs ~{1e3*p1['P_phys']/mi:.0f} ns per worker "
              f"(bignum multiply + mask + ctypes store), while a coherence line transfer costs on the order of 100 ns and is partly "
              f"overlapped by the store buffer, so the interpreter hides the false-sharing penalty. The expected boundary is the "
              f"{line} B line (stride {line//8}); in compiled C the same test shows a large drop there. ")
        w(f"Karp-Flatt e(p) for stride=1: {', '.join(f(x,3) for x in e1) or '-'}; for stride=16: "
          f"{', '.join(f(x,3) for x in e16) or '-'}. e(p) rising with p means overhead that grows with p (process spawn, "
          f"coherence traffic, slower cores joining) rather than a fixed serial part; the lower e(p) at stride=16 is the part "
          f"attributable to line sharing.\n")
    if p2:
        st, wk = p2["strong"], p2["weak"]
        w(f"**Phase 2.** S_strong(P_log={st[-1]['p']}) = {st[-1]['S']:.2f} vs Amdahl cap {p2['amdahl_cap']:.2f}: "
          f"the cap is far away because s = {p2['s']:.4f}; the real limit is IPC/pool dispatch overhead, shared cores and turbo drop "
          f"under all-core load. Weak scaling: E_weak(P_log) = {wk[-1]['Ew']:.2f}, Gustafson S = {wk[-1]['SG']:.2f} "
          f"(α = {wk[-1]['alpha']:.4f}) — grows almost linearly because the parallel part grows with p.")
        b = p2["P_phys"] if hw["P_log"] > p2["P_phys"] else (hw.get("P_cores") if hw.get("P_cores") and hw["P_cores"] < hw["P_log"] else None)
        if b:
            S = {r["p"]: r["S"] for r in st}
            kind = "SMT" if hw["P_log"] > p2["P_phys"] else "P-core → E-core"
            db, da = S[b] - S.get(b - 1, 0.0), S[b + 1] - S[b]
            dS = {q: S[q] - S[q - 1] for q in range(2, max(S) + 1)}
            w(f" Marginal ΔS(p−1→p): " + ", ".join(f"{q}: {v:+.2f}" for q, v in dS.items()) + ".")
            if da < db:
                w(f" Inflection ({kind} boundary at p = {b}): ΔS = {db:+.3f} for p={b-1}→{b} vs {da:+.3f} for p={b}→{b+1} "
                  f"({100*(1-da/db):.0f}% drop) — the extra worker lands on a slower core sharing resources.\n")
            else:
                after = [q for q in dS if q > b + 1]
                q2 = max(after, key=lambda q: dS[q-1] - dS[q]) if after else b + 1
                w(f" At the {kind} boundary (p = {b}) the drop is NOT at {b}→{b+1}: ΔS = {db:+.3f} for {b-1}→{b} but {da:+.3f} for "
                  f"{b}→{b+1}. With p = {b} the {b} performance cores are shared with the parent process (serial Stage A + Pool "
                  f"dispatch/IPC), so p = {b} under-performs; the {b+1}th worker is absorbed by an efficiency core. Beyond p = {b+1} "
                  f"every new worker runs on an E-core (lower clock/IPC) and ΔS falls to {dS[max(dS)]:+.2f} at p = {max(dS)} "
                  f"(largest step down at p = {q2}). macOS does not pin processes to cores, so the boundary is blurred by the scheduler.\n")
        else:
            w(" P_log = P_phys and no hybrid cores reported, so there is no SMT / E-core step to measure.\n")
    if p3:
        g = {(x["variant"], x["BS"]): x for x in p3["gemm"]}; bs = p3["BS_star"]; b = g[(3, bs)]
        l1, l2 = hw["L1d"], hw["L2"]
        where = ("fits in L1d" if l1 and b["W_tile_B"] <= l1 else "fits in L2" if l2 and b["W_tile_B"] <= l2 else "exceeds L2")
        w(f"**Phase 3.** BW = {p3['bw_gbs']:.2f} GB/s, P = {p3['peak_gflops']:.4f} GFLOP/s → I* = {p3['ridge']:.5f} FLOP/B. "
          f"BS* = {bs}: W_tile = {kib(b['W_tile_B'])} ({where}; L1d = {kib(l1)}, L2 = {kib(l2)}), I_tiled ≈ {b['I_tiled']:.2f} FLOP/B. "
          f"Variant 1 {g[(1,None)]['gflops']:.4f}, Variant 2 {g[(2,None)]['gflops']:.4f}, Variant 3 {b['gflops']:.4f} GFLOP/s "
          f"(= {100*b['gflops']/p3['peak_gflops']:.0f}% of P_sustained). On the roofline all three variants lie to the right of the ridge "
          f"(I_naive ≈ 0.125, I_ikj ≈ 0.125 but unit-stride, I_tiled ≈ BS/12 ≫ I*), i.e. they are compute-bound by the CPython interpreter, "
          f"not by DRAM. Therefore loop order/tiling changes performance only by a few % here; in compiled code the same tiling gives multiples.")
        best_v = max((x for x in p3["gemm"]), key=lambda x: x["gflops"])
        if best_v["variant"] == 1:
            w(" Variant 1 is the fastest in CPython because its inner loop keeps the sum in a local variable (1 memoryview read of A, "
              "1 of B per step), whereas Variants 2/3 do `C[i*N+j] += ...` — an extra memoryview read + write per step; "
              "interpreter op count outweighs the cache benefit of unit stride. Tiling (Variant 3) recovers part of the loss as BS grows "
              "because loop-control overhead per FMA shrinks.")
        w("\n")
    if p4:
        rows = {(r["strategy"], r["C"]): r for r in p4["rows"]}; bl = rows[("Block static", None)]
        d = {c: rows[("Dynamic", c)] for c in (1, 8, 64, 512)}; cs = p4["C_star"]
        w(f"**Phase 4.** Block static: T_wall = {bl['T_wall_s']:.3f} s, λ = {bl['lambda']:.2f} — the last block holds most of the "
          f"K(i) ∝ i² work. Dynamic C=1: λ = {d[1]['lambda']:.3f}, T = {d[1]['T_wall_s']:.3f} s — best balance but one lock "
          f"acquisition per item (synchronization contention). C=512: λ = {d[512]['lambda']:.3f}, T = {d[512]['T_wall_s']:.3f} s — "
          f"few lock ops but the final heavy chunks create tail imbalance (N/C = {p4['N']//512} chunks for {p4['p']} workers). "
          f"Optimal C* = {cs} (T = {d[cs]['T_wall_s']:.3f} s, speedup ×{bl['T_wall_s']/d[cs]['T_wall_s']:.2f} over block static).")
        if cs <= 8:
            avgk = p4["total_iters"] / p4["N"]
            w(f" Small C wins here because each item is heavy (mean K = {avgk:,.0f} trig iterations ≈ milliseconds), while one "
              f"lock-protected counter update costs only microseconds, so contention overhead is < 1% and perfect balance dominates.")
        w("\n")
    w("## 4. Scripts\n")
    for fn in ["harness.py", "common.py", "task0_cal.py", "task1_falsesharing.py", "task2_scaling.py",
               "task3_roofline_gemm.py", "task4_scheduling.py", "make_report.py", "run_all.py"]:
        p = Path(__file__).with_name(fn)
        if p.exists():
            w(f"### {fn}\n```python\n{p.read_text().rstrip()}\n```\n")
    out = Path(__file__).with_name(f"lab_report_{STUDENT_ID}.md")
    out.write_text("\n".join(L))
    print(f"Report written: {out}")

if __name__ == "__main__":
    main()
