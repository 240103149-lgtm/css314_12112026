"""Usage:  python run_all.py <STUDENT_ID> [--quick] [--resume]
Runs Phases 0-4 in order and writes lab_report_<STUDENT_ID>.md next to this file."""
import os, subprocess, sys, shutil, time
from pathlib import Path

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or not args[0].isdigit():
        sys.exit("Usage: python run_all.py <STUDENT_ID> [--quick]")
    env = dict(os.environ, STUDENT_ID=args[0], LAB_QUICK="1" if "--quick" in sys.argv else "0")
    here = Path(__file__).parent
    resume = "--resume" in sys.argv
    if not resume:
        shutil.rmtree(here / "results", ignore_errors=True)
    for t in ["task0_cal", "task1_falsesharing", "task2_scaling", "task3_roofline_gemm", "task4_scheduling", "make_report"]:
        if resume and t.startswith("task") and (here / "results" / f"phase{t[4]}.json").exists():
            print(f"skip {t} (results exist)"); continue
        print(f"\n===== {t} =====", flush=True); t0 = time.time()
        subprocess.run([sys.executable, str(here / f"{t}.py")], env=env, cwd=here, check=True)
        print(f"({time.time() - t0:.0f} s)", flush=True)

if __name__ == "__main__":
    main()
