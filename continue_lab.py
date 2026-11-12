"""Continue Phases 3-4 using standard Python; preserve user-supplied Phase 2."""
from __future__ import annotations
import argparse
import atexit
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SID = 240103149
STATE = HERE / 'continuation_results.json'
REPORT = HERE / f'lab_report_{SID}.md'
RAW = HERE / 'continuation_telemetry.txt'
TILES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
D = {}


def log(message):
    print(message, flush=True)
    with RAW.open('a', encoding='utf-8') as f:
        f.write(str(message) + '\n')


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(D, indent=2), encoding='utf-8')
    temp.replace(STATE)


def verify(phase, a, b, logical):
    payload = f'{phase}|{SID}|{logical}|{a:.4f}|{b:.4f}'
    h = 14695981039346656037
    for byte in payload.encode('utf-8'):
        h = ((h ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f'VERIFY_HASH [{phase}]: {h:016X} (P_log={logical})'


def bench(fn, warmups, runs):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    xs = sorted(samples)
    return dict(median_ms=statistics.median(xs),
                iqr_ms=xs[3*len(xs)//4] - xs[len(xs)//4], samples_ms=samples)


def sysctl(name):
    try:
        return subprocess.check_output(['sysctl', '-n', name], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def hardware():
    hw = {'OS': platform.platform(), 'Python': sys.version.split()[0],
          'logical_cpus': mp.cpu_count()}
    for key in ('machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.cachelinesize',
                'hw.l1dcachesize', 'hw.l2cachesize', 'hw.l3cachesize',
                'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu',
                'hw.perflevel0.l1dcachesize', 'hw.perflevel1.l1dcachesize',
                'hw.perflevel0.l2cachesize', 'hw.perflevel1.l2cachesize'):
        hw[key] = sysctl(key)
    return hw


_HANDLES = []
_VIEWS = []
_N = 0


def close_views():
    for view in _VIEWS:
        view.release()
    _VIEWS.clear()
    for handle in _HANDLES:
        handle.close()
    _HANDLES.clear()


def attach(names, fmt, n=0):
    global _N
    _N = n
    for name in names:
        handle = shared_memory.SharedMemory(name=name)
        _HANDLES.append(handle)
        _VIEWS.append(handle.buf.cast(fmt))
    atexit.register(close_views)


def ready(_):
    return True


def copy_job(bounds):
    lo, hi = bounds
    _VIEWS[0][lo:hi] = _VIEWS[1][lo:hi]


def compute_job(iterations):
    x0=x1=x2=x3=x4=x5=x6=x7=x8=x9=x10=x11=x12=x13=x14=x15=0.1
    a, b = 1.0000001, 0.0000003
    for _ in range(iterations):
        x0=x0*a+b; x1=x1*a+b; x2=x2*a+b; x3=x3*a+b
        x4=x4*a+b; x5=x5*a+b; x6=x6*a+b; x7=x7*a+b
        x8=x8*a+b; x9=x9*a+b; x10=x10*a+b; x11=x11*a+b
        x12=x12*a+b; x13=x13*a+b; x14=x14*a+b; x15=x15*a+b
    return sum((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15))


def gemm_job(args):
    lo, hi, variant, bs = args
    A, B, C = _VIEWS
    n = _N
    # Clearing output is included consistently in all GEMM timings.
    for pos in range(lo*n, hi*n):
        C[pos] = 0.0
    if variant == 'naive':
        for i in range(lo, hi):
            row = i*n
            for j in range(n):
                dot = 0.0
                for k in range(n):
                    dot += A[row+k] * B[k*n+j]
                C[row+j] = dot
    elif variant == 'reordered':
        for i in range(lo, hi):
            row = i*n
            for k in range(n):
                aik, brow = A[row+k], k*n
                for j in range(n):
                    C[row+j] += aik * B[brow+j]
    elif variant == 'blocked':
        for ii in range(lo, hi, bs):
            for jj in range(0, n, bs):
                endj = min(n, jj+bs)
                for kk in range(0, n, bs):
                    for i in range(ii, min(hi, ii+bs)):
                        # Native contiguous views hoisted outside the j loop.
                        crow = C[i*n+jj:i*n+endj]
                        for k in range(kk, min(n, kk+bs)):
                            aik = A[i*n+k]
                            brow = B[k*n+jj:k*n+endj]
                            for j in range(endj-jj):
                                crow[j] += aik * brow[j]
                            brow.release()
                        crow.release()
    else:
        raise ValueError(variant)
    return True


def ranges(n, p):
    return [(t*n//p, (t+1)*n//p) for t in range(p)]


def phase3(p, warmups, runs):
    r = D.setdefault('phase3', {})
    size = 96_000_000
    if 'bandwidth' not in r:
        buffers = [shared_memory.SharedMemory(create=True, size=size) for _ in range(2)]
        try:
            # First touch every page before timing. The source is nonzero.
            for index, handle in enumerate(buffers):
                block = bytes([index+1]) * 1_000_000
                for offset in range(0, size, len(block)):
                    handle.buf[offset:offset+len(block)] = block
            with mp.Pool(p, initializer=attach,
                         initargs=([h.name for h in buffers], 'B')) as pool:
                pool.map(ready, range(p), chunksize=1)
                log('Phase 3A: DRAM copy (96 MB read + 96 MB write)')
                stats = bench(lambda: pool.map(copy_job, ranges(size,p), chunksize=1), warmups,runs)
            stats['GBps'] = 2*size/(stats['median_ms']*1e6)
            if bytes(buffers[0].buf[:64]) != bytes(buffers[1].buf[:64]):
                raise RuntimeError('Memory copy validation failed')
            r['bandwidth'] = stats
            save()
        finally:
            for h in buffers:
                h.close(); h.unlink()
    if 'compute' not in r:
        with mp.Pool(p) as pool:
            pool.map(ready, range(p), chunksize=1)
            log('Phase 3A: 80,000,000 FLOPs per physical worker')
            stats = bench(lambda:pool.map(compute_job,[2_500_000]*p,chunksize=1),warmups,runs)
        stats['GFLOPs'] = p*80_000_000/(stats['median_ms']*1e6)
        r['compute'] = stats
        save()
    n = 384
    buffers = [shared_memory.SharedMemory(create=True, size=n*n*8) for _ in range(3)]
    views = [h.buf.cast('d') for h in buffers]
    try:
        for i in range(n*n):
            views[0][i] = (i % 17 + 1)/17
            views[1][i] = (i % 13 + 1)/13
        rows = r.setdefault('gemm', {})
        with mp.Pool(p, initializer=attach, initargs=([h.name for h in buffers],'d',n)) as pool:
            pool.map(ready, range(p), chunksize=1)
            def run(variant, bs=0):
                pool.map(gemm_job,[(lo,hi,variant,bs) for lo,hi in ranges(n,p)],chunksize=1)
            # Untimed reference; all variants use the same ascending k order.
            run('reordered')
            reference = list(views[2])
            jobs = [('naive',0),('reordered',0)] + [('blocked',b) for b in TILES]
            for variant, bs in jobs:
                key = variant if not bs else f'blocked_{bs}'
                if key in rows:
                    continue
                log(f'Phase 3B: {key} ({warmups} warmups + {runs} timed)')
                stats = bench(lambda:run(variant,bs),warmups,runs)
                error = max(abs(a-b) for a,b in zip(reference,views[2]))
                if error >= 1e-9:
                    raise RuntimeError(f'{key}: maximum error {error} exceeds tolerance')
                stats.update(tile=bs, GFLOPs=2*n**3/(stats['median_ms']*1e6), max_abs_error=error)
                rows[key] = stats
                save()
                write_report()
        best = min((k for k in rows if k.startswith('blocked_')), key=lambda k:rows[k]['median_ms'])
        r['best'] = best
        r['ridge'] = r['compute']['GFLOPs']/r['bandwidth']['GBps']
        r['verify'] = verify('PHASE3',rows['naive']['GFLOPs'],rows[best]['GFLOPs'],mp.cpu_count())
        log(r['verify'])
        save()
    finally:
        for view in views:
            view.release()
        for h in buffers:
            h.close(); h.unlink()
    write_report()


def item_work(i):
    x = (i+1)*1e-5
    for _ in range(1+i*i//3600):
        x = math.sin(x)+math.cos(x)+math.sqrt(abs(x)+1e-12)
    return x


def scheduling_worker(t,p,n,strategy,chunk,counter,elapsed,checksums,barrier):
    barrier.wait()
    start = time.perf_counter_ns()
    checksum = 0.0
    if strategy == 'dynamic':
        while True:
            with counter.get_lock():
                lo = counter.value
                counter.value += chunk
            if lo >= n:
                break
            for i in range(lo,min(n,lo+chunk)):
                checksum += item_work(i)
    else:
        items = range(t*n//p,(t+1)*n//p) if strategy == 'block' else range(t,n,p)
        for i in items:
            checksum += item_work(i)
    elapsed[t] = (time.perf_counter_ns()-start)/1e6
    checksums[t] = checksum


def schedule(p,strategy,chunk,n=12000):
    counter = mp.Value('i',0)
    elapsed, checksums = mp.RawArray('d',p),mp.RawArray('d',p)
    barrier = mp.Barrier(p)
    workers = [mp.Process(target=scheduling_worker,
                         args=(t,p,n,strategy,chunk,counter,elapsed,checksums,barrier)) for t in range(p)]
    start = time.perf_counter_ns()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError('Scheduling worker failed')
    wall = (time.perf_counter_ns()-start)/1e6
    times = list(elapsed)
    return dict(wall_ms=wall,worker_ms=times,imbalance=max(times)/statistics.mean(times),
                checksum=sum(checksums),chunk=chunk)


def phase4(p):
    rows = D.setdefault('phase4',{}).setdefault('rows',{})
    for strategy,chunk in [('block',0),('cyclic',0)]+[('dynamic',c) for c in (1,4,16,64,256,1024)]:
        key = strategy if not chunk else f'dynamic_{chunk}'
        if key in rows:
            continue
        log(f'Phase 4: {key}; N=12,000; one complete measured run')
        result = schedule(p,strategy,chunk)
        if rows:
            reference = next(iter(rows.values()))['checksum']
            if not math.isclose(result['checksum'],reference,rel_tol=1e-10,abs_tol=1e-8):
                raise RuntimeError('Scheduler checksum differs; missing or duplicated work')
        rows[key] = result
        log(f"{key}: wall_ms={result['wall_ms']:.4f}; lambda={result['imbalance']:.6f}; worker_ms={result['worker_ms']}")
        save()
        write_report()
    best = min((k for k in rows if k.startswith('dynamic_')),key=lambda k:rows[k]['wall_ms'])
    r = D['phase4']
    r['best'] = best
    r['verify'] = verify('PHASE4',rows['block']['wall_ms'],rows[best]['wall_ms'],mp.cpu_count())
    log(r['verify'])
    save()
    write_report()


def capture_previous():
    # Only lab-named text and source files immediately beside this script.
    found = {}
    for pattern in ('task*.py','harness.py','phase*.txt','*telemetry*.txt',f'lab_report_{SID}.md'):
        for path in HERE.glob(pattern):
            if path.name in (REPORT.name,RAW.name) or path.stat().st_size > 2_000_000:
                continue
            found[path.name] = path.read_text(encoding='utf-8',errors='replace')
    if REPORT.exists() and not REPORT.read_text(encoding='utf-8').startswith('# Parallel Computing Lab - continuation'):
        found['earlier_'+REPORT.name] = REPORT.read_text(encoding='utf-8',errors='replace')
    if found:
        D.setdefault('previous_files',{}).update(found)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def write_report():
    phase2 = json.loads((HERE/'phase2_results.json').read_text())
    lines = ['# Parallel Computing Lab - continuation',f'\nStudent ID: {SID}',
             '\nPhase 2 comes from the supplied terminal summary. Phases 3-4 below are populated only from local measurements. This report is incomplete until prior Phase 0-1 evidence and source code are included.',
             '\n## Hardware and previous phases']
    hw = D.get('hardware',{})
    if hw:
        lines.append(table(['Parameter','Measured OS value'],hw.items()))
    else:
        lines.append('Hardware audit pending local execution.')
    previous = D.get('previous_files',{})
    for phase in ('PHASE0','PHASE1'):
        matches = [line for text in previous.values() for line in text.splitlines() if 'VERIFY_HASH' in line and phase in line]
        lines.append('\n'+ ('\n'.join(dict.fromkeys(matches)) if matches else f'{phase}: prior measurements and verification hash not supplied.'))
    lines += ['\n## Phase 2: Strong scaling', 'N = 1,600,000; 2 warmups + 7 timed runs; 35 parallel iterations per element; serial prefix = 5% of elements.']
    strong = phase2['strong']; weak = phase2['weak']; t1 = strong[0][1]
    speed = [t1/r[1] for r in strong]
    lines.append(table(['p','N','Median ms','IQR ms','Speedup','Efficiency'],
                       [[p,1600000,f'{m:.4f}',f'{iq:.4f}',f'{t1/m:.4f}',f'{t1/m/p:.4f}'] for p,m,iq in strong]))
    lines.append('\n### Weak scaling')
    lines.append(table(['p','N(p)','Median ms','IQR ms','Efficiency','Alpha','Gustafson S'],
                       [[p,n,f'{m:.4f}',f'{iq:.4f}',f'{weak[0][2]/m:.4f}',f'{a:.6f}',f'{p-a*(p-1):.4f}'] for p,n,m,iq,a in weak]))
    s = phase2['stage_a_strong_p1_ms']/t1
    theory = 1/(s+(1-s)/8)
    observed = speed[-1]
    lines.append(f'\nMeasured serial fraction s = 31.9643 / 2546.9937 = {s:.8f}. Amdahl S(infinity) = {1/s:.4f}; S(8) = {theory:.4f}; observed S(8) = {observed:.4f}. Observed speedup is {100*(1-observed/theory):.2f}% below this ideal model. Pool dispatch, memory effects, interpreter execution and heterogeneous CPU resources can contribute; these measurements alone do not isolate their individual costs. A 5% element prefix is not a 5% runtime fraction because the stages perform different work per element.')
    gains = [speed[i]-speed[i-1] for i in range(1,8)]
    lines.append(table(['Transition','Added speedup','Change vs prior gain %'],
                       [[f'{i+1} -> {i+2}',f'{g:.4f}','N/A' if i==0 else f'{100*(g/gains[i-1]-1):.2f}'] for i,g in enumerate(gains)]))
    drop = 100*(1-gains[3]/gains[2])
    lines.append(f'\nThe marginal gain falls from {gains[2]:.4f} at 3->4 workers to {gains[3]:.4f} at 4->5, a {drop:.2f}% decrease. This is an observed inflection, not proof of a particular hardware boundary. On a confirmed 4-performance/4-efficiency-core machine it is consistent with core heterogeneity, but OS scheduling does not guarantee workers occupy performance cores first. If physical and logical core counts are equal, there is no SMT boundary in this sweep.')
    alpha = phase2['stage_a_weak_p8_ms']/weak[-1][2]
    lines.append(f'\nAt p=8, alpha = 45.9141 / 804.6131 = {alpha:.8f}; Gustafson S = 8-alpha*7 = {8-alpha*7:.4f}. This is a modeled scaled speedup, not an independently measured same-size serial/parallel speedup. Weak efficiency is {weak[0][2]/weak[-1][2]:.4f}. The weak serial stage grows with total N, so fixed work per worker does not keep the serial prefix constant. Weak-scaling p=6 IQR/median = {100*weak[5][3]/weak[5][2]:.2f}%, indicating appreciable run variability.')
    lines.append('```text\n'+phase2['verify']+'\n```')
    lines.append('The Phase 2 hash recomputes correctly from PHASE2|240103149|8|4.1114|7.6006. Original sub-sample timings were not supplied; derived values use the reported rounded summary.')
    r = D.get('phase3',{})
    lines.append('\n## Phase 3: Empirical roofline and cache tiling')
    if 'compute' in r and 'bandwidth' in r:
        bw,compute = r['bandwidth']['GBps'],r['compute']['GFLOPs']
        lines.append(f'\nSustained bandwidth = {bw:.8f} GB/s; Python compute ceiling = {compute:.8f} GFLOP/s; ridge point = {compute/bw:.8f} FLOP/byte. FLOPs count a multiply and an add separately (two operations); this Python expression does not imply a native fused FMA instruction. The 192 MB copy footprint is first-touched before timing. Persistent worker pools exclude process startup; pool dispatch and completion are timed. Repeated copies may benefit from platform caches, so this is an effective streaming-copy estimate, not a direct memory-controller traffic measurement.')
        rows = r.get('gemm',{})
        lines.append(table(['Variant','Tile','Median ms','IQR ms','GFLOP/s','Tile bytes','Max error'],
                           [[key,v['tile'] or '-',f"{v['median_ms']:.4f}",f"{v['iqr_ms']:.4f}",f"{v['GFLOPs']:.8f}",24*v['tile']**2 if v['tile'] else '-',f"{v['max_abs_error']:.3e}"] for key,v in rows.items()]))
        if 'best' in r:
            best = rows[r['best']]; bs = best['tile']; footprint = 24*bs*bs
            lines.append(f'\nBest tested tile BS* = {bs}; tile footprint = {footprint} bytes ({footprint/1024:.2f} KiB). Compare this with the measured cache capacities above, including separate performance levels when available. Three 384x384 double matrices occupy 3,538,944 bytes; whether this exceeds L2 depends on the actual processor. Tile choice reflects Python loop/slice overhead and row partition boundaries as well as cache capacity; do not infer cache size from the optimum alone.')
            roofrows = []
            for key,intensity,note in [('naive',.125,'Analytical estimate'),('reordered',.125,'Conservative naive traffic proxy; actual reuse not measured'),(r['best'],bs/12,'Ideal tile-reuse estimate')]:
                achieved = rows[key]['GFLOPs']; attainable = min(compute,bw*intensity)
                roofrows.append([key,f'{intensity:.6f}',f'{attainable:.8f}',f'{achieved:.8f}',f'{100*achieved/attainable:.2f}',note])
            lines.append(table(['Variant','Intensity FLOP/B','Roof GFLOP/s','Achieved GFLOP/s','Roof %','Assumption'],roofrows))
            lines.append('Roof = min(P_sustained, BW_sustained * intensity). Python GEMM uses indexing, object operations and stores absent from the local-scalar compute kernel; these empirical ceilings are proxies. Intensity is analytical rather than measured with hardware counters. Percentages above 100%, if observed, indicate model/workload mismatch and must not be clipped.')
            lines.append('```text\n'+r['verify']+'\n```')
    else:
        lines.append('Pending local measurements. No benchmark numbers have been invented.')
    lines.append('\n## Phase 4: Scheduling and load imbalance')
    r = D.get('phase4',{}); rows = r.get('rows',{})
    if rows:
        lines.append(table(['Strategy','Chunk','Wall ms','Lambda','Worker active times ms'],
                           [[key,v['chunk'] or '-',f"{v['wall_ms']:.4f}",f"{v['imbalance']:.6f}",', '.join(f'{x:.3f}' for x in v['worker_ms'])] for key,v in rows.items()]))
        lines.append('N=12,000; K(i)=1+floor(i*i/3600). Every strategy processes the same items; aggregate checksums are validated. One measured complete run per setting; no IQR or statistical superiority claim is possible. Wall time includes process creation, synchronization and joins. Active worker time starts after the common barrier and includes counter-lock waits in dynamic mode. Lambda=max(worker times)/mean(worker times).')
        if 'best' in r:
            best = rows[r['best']]; ratio = rows['block']['wall_ms']/best['wall_ms']
            lines.append(f"\nBest measured dynamic chunk C*={best['chunk']}; static-block/dynamic wall-time ratio = {ratio:.4f}. Contiguous blocks assign the most expensive tail items to the last worker; cyclic assignment spreads those items. C=1 makes many shared-counter lock acquisitions but minimizes chunk-tail skew; C=1024 reduces acquisitions but can leave a costly final chunk. These are mechanisms, not a guarantee of monotonic timing. A hybrid CPU can make dynamic assignment useful even where equal item counts are statically balanced. Lambda measures time balance and does not directly measure CPU utilization. This is a centralized shared-counter queue, not per-worker deque stealing.")
            lines.append('```text\n'+r['verify']+'\n```')
    else:
        lines.append('Pending local measurements.')
    lines.append('\n## Completeness and evidence')
    text = '\n'.join(previous.values())
    for phase in ('PHASE0','PHASE1'):
        lines.append(f"- {phase} evidence: "+('found in prior files; inspect appendix for required tables and analyses.' if phase in text else 'MISSING: provide original report/log and source.'))
    lines.append('- Phase 2 original source: '+('found; see appendix.' if any('task2' in key for key in previous) else 'not supplied; retain your existing task2_scaling.py alongside this file and rebuild.'))
    lines.append('- Phase 3: '+('complete.' if 'verify' in D.get('phase3',{}) else 'pending.'))
    lines.append('- Phase 4: '+('complete.' if 'verify' in D.get('phase4',{}) else 'pending.'))
    if RAW.exists():
        lines += ['\n## Continuation raw telemetry','```text',RAW.read_text(encoding='utf-8'),'```']
    for name,text in sorted(previous.items()):
        lines += ['\n## Prior evidence: '+name,'````'+('python' if name.endswith('.py') else 'text'),text,'````']
    lines += ['\n## Source appendix: continue_lab.py','````python',Path(__file__).read_text(encoding='utf-8'),'````']
    temp = REPORT.with_suffix('.tmp')
    temp.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    temp.replace(REPORT)


def self_test():
    n,p = 12,2
    buffers = [shared_memory.SharedMemory(create=True,size=n*n*8) for _ in range(3)]
    views = [h.buf.cast('d') for h in buffers]
    try:
        for i in range(n*n):
            views[0][i]=(i%7)/7; views[1][i]=(i%11)/11
        ref = [sum(views[0][i*n+k]*views[1][k*n+j] for k in range(n)) for i in range(n) for j in range(n)]
        with mp.Pool(p,initializer=attach,initargs=([h.name for h in buffers],'d',n)) as pool:
            for variant,bs in [('naive',0),('reordered',0),('blocked',4),('blocked',8),('blocked',128)]:
                pool.map(gemm_job,[(lo,hi,variant,bs) for lo,hi in ranges(n,p)])
                assert max(abs(a-b) for a,b in zip(ref,views[2])) < 1e-9
        reference = None
        for kind,chunk in [('block',0),('cyclic',0),('dynamic',1),('dynamic',16)]:
            result = schedule(p,kind,chunk,n=120)
            if reference is not None:
                assert math.isclose(reference,result['checksum'],rel_tol=1e-10)
            reference = result['checksum']
        print('SELF TEST PASSED: GEMM variants, partial tiles, schedulers and checksums.')
    finally:
        for v in views: v.release()
        for h in buffers: h.close(); h.unlink()


def main():
    global D
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['remaining','3','4','report','self-test'],nargs='?',default='remaining')
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--runs',type=int,default=7)
    args = parser.parse_args()
    if args.phase == 'self-test':
        self_test(); return
    if args.warmups < 0 or args.runs < 1:
        parser.error('warmups must be >=0; runs must be >=1')
    D = json.loads(STATE.read_text()) if STATE.exists() else {}
    capture_previous()
    if args.phase == 'report':
        save(); write_report(); print(REPORT); return
    hw = hardware()
    if hw['logical_cpus'] != 8:
        raise SystemExit('This continuation belongs to the supplied 8-logical-CPU machine. Run it on that Mac; results from another machine cannot be mixed.')
    physical = hw.get('hw.physicalcpu','unavailable')
    if not str(physical).isdigit():
        raise SystemExit('Physical core count unavailable. Run this package on the original macOS laptop.')
    config = {'physical_workers':int(physical),'warmups':args.warmups,'timed_runs_phase3':args.runs,'phase4_runs':1}
    if D.get('config') and D['config'] != config:
        raise SystemExit('Measurement settings differ from saved results. Keep this run consistent or start in a new folder.')
    if D.get('hardware') and D['hardware'] != hw:
        raise SystemExit('Hardware/Python/OS differs from saved results. Do not mix runs from different environments.')
    D['config'],D['hardware'] = config,hw
    save(); write_report()
    log(f'Student {SID}; physical workers={physical}; logical CPUs=8; spawn multiprocessing')
    log('Saved settings are resumed; completed measurements are not repeated.')
    if args.phase in ('remaining','3'): phase3(int(physical),args.warmups,args.runs)
    if args.phase in ('remaining','4'): phase4(int(physical))
    log(f'REPORT={REPORT}')
    write_report()
    print('Finished. Open '+REPORT.name)


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn',force=True)
    main()
