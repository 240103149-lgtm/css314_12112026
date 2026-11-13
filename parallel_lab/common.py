"""Shared config, hardware detection and result persistence for all phases."""
import json, os, platform, re, subprocess, multiprocessing as mp
from pathlib import Path

STUDENT_ID = int(os.environ.get("STUDENT_ID", "0"))
QUICK = os.environ.get("LAB_QUICK", "0") == "1"   # smoke-test mode: smaller sizes, fewer runs
RESULTS = Path(__file__).with_name("results")
RESULTS.mkdir(exist_ok=True)

def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""

def _to_bytes(txt):
    m = re.search(r"([\d.]+)\s*([KMG]i?B?)?", txt or "")
    if not m: return None
    v = float(m.group(1)); u = (m.group(2) or "").upper()[:1]
    return int(v * {"K": 1024, "M": 1024**2, "G": 1024**3}.get(u, 1))

def hw_info():
    cache = RESULTS / "hw.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sysname = platform.system()
    info = {"os": f"{sysname} {platform.release()}", "python": platform.python_version(),
            "P_log": mp.cpu_count(), "P_phys": None, "model": platform.processor(),
            "line": 64, "L1d": None, "L2": None, "L3": None, "raw": ""}
    if sysname == "Linux":
        raw = _sh('lscpu | grep -E "Model name|Core|Thread|Socket|L1d|L2|L3"')
        info["raw"] = raw
        kv = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in raw.splitlines() if ":" in l}
        info["model"] = kv.get("Model name", info["model"])
        try: info["P_phys"] = int(kv["Core(s) per socket"]) * int(kv.get("Socket(s)", "1"))
        except Exception: pass
        for k in ("L1d", "L2", "L3"):
            key = next((x for x in kv if x.startswith(k)), None)
            if key:
                total = _to_bytes(kv[key]); n = re.search(r"\((\d+) instance", kv[key])
                info[k] = total // int(n.group(1)) if (total and n and k != "L3") else total
        ln = _sh("getconf LEVEL1_DCACHE_LINESIZE")
        if ln.isdigit() and int(ln) > 0: info["line"] = int(ln)
    elif sysname == "Darwin":
        info["raw"] = _sh("sysctl -n machdep.cpu.brand_string && sysctl hw.physicalcpu hw.logicalcpu "
                          "hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize "
                          "hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu")
        g = lambda k: _sh(f"sysctl -n {k}")
        info["model"] = g("machdep.cpu.brand_string") or info["model"]
        for key, k in (("P_phys", "hw.physicalcpu"), ("line", "hw.cachelinesize"),
                       ("L1d", "hw.l1dcachesize"), ("L2", "hw.l2cachesize"), ("L3", "hw.l3cachesize"),
                       ("P_cores", "hw.perflevel0.physicalcpu"), ("E_cores", "hw.perflevel1.physicalcpu")):
            v = g(k)
            if v.isdigit(): info[key] = int(v)
    elif sysname == "Windows":
        raw = _sh('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select Name, '
                  'NumberOfCores, NumberOfLogicalProcessors, L2CacheSize, L3CacheSize | Format-List"')
        info["raw"] = raw
        kv = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in raw.splitlines() if ":" in l}
        info["model"] = kv.get("Name", info["model"])
        if kv.get("NumberOfCores", "").isdigit(): info["P_phys"] = int(kv["NumberOfCores"])
        if kv.get("L2CacheSize", "").isdigit(): info["L2"] = int(kv["L2CacheSize"]) * 1024
        if kv.get("L3CacheSize", "").isdigit(): info["L3"] = int(kv["L3CacheSize"]) * 1024
    if os.environ.get("P_PHYS"): info["P_phys"] = int(os.environ["P_PHYS"])
    if not info["P_phys"]: info["P_phys"] = info["P_log"]
    cache.write_text(json.dumps(info, indent=2))
    return info

def save(phase, data):
    (RESULTS / f"{phase}.json").write_text(json.dumps(data, indent=2))

def load(phase):
    p = RESULTS / f"{phase}.json"
    return json.loads(p.read_text()) if p.exists() else None
