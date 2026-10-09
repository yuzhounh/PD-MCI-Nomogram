import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time


PIPELINE_MANIFEST = [
    ("Python", "step_01_extract_data.py"),
    ("Python", "step_02_initial_missing_stats.py"),
    ("Python", "step_04_process_features.py"),
    ("Python", "step_05_split_data.py"),
    ("Python", "step_06_calculate_duration.py"),
    ("Python", "step_07_missing_stats.py"),
    ("Python", "step_08_imputation.py"),
    ("Python", "step_09_characteristics_stats.py"),
    ("Python", "step_10_correlation_analysis.py"),
    ("Python", "step_11_remove_columns.py"),
    ("Python", "step_12_lasso.py"),
    ("Python", "step_13_lr.py"),
    ("Python", "step_14_catboost.py"),
    ("Python", "step_14_easyensemble.py"),
    ("Python", "step_14_lightgbm.py"),
    ("Python", "step_14_lr.py"),
    ("Python", "step_14_mlp.py"),
    ("Python", "step_14_rf.py"),
    ("Python", "step_14_svm.py"),
    ("Python", "step_14_xgboost.py"),
    ("Python", "step_14_bootstrap_ci.py"),
    ("Python", "step_15_plot_roc_pr_curves.py"),
    ("R", "step_16_nomogram.R"),
]


def extract_step_num(filename):
    """Extract the step number from a filename for sorting."""
    base = os.path.basename(filename)
    m = re.search(r"step_(\d+)", base, re.IGNORECASE)
    if m:
        return (int(m.group(1)), base)
    return (9999, base)


def get_rscript_path():
    """Locate the Rscript executable."""
    # 1. Search PATH first.
    rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if rscript:
        return rscript

    # 2. Search common Windows installation locations.
    candidates = [
        r"C:\Program Files\R\R-*\bin\Rscript.exe",
        r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
        r"C:\Program Files (x86)\R\R-*\bin\Rscript.exe",
        r"C:\ProgramData\Anaconda3\Scripts\Rscript.exe",
        r"C:\ProgramData\Anaconda3\envs\*\Scripts\Rscript.exe",
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(reverse=True)
            return matches[0]

    return "Rscript"


def run_script(script, runner_cmd, extra_args=None):
    """Run one script and capture its output."""
    cmd = [runner_cmd, script]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\nExecuting command: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start_time

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"[OK] {script} completed successfully ({elapsed:.2f}s)")
            return True
        else:
            print(
                f"[FAIL] {script} failed with return code {result.returncode} ({elapsed:.2f}s)"
            )
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Error executing {script}: {str(e)} ({elapsed:.2f}s)")
        return False


def main():
    # Use the script directory as the working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="Run all step scripts in sequence by numerical step order."
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Run Python step scripts only",
    )
    parser.add_argument(
        "--r-only",
        action="store_true",
        help="Run R step scripts only",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of initial scripts to skip (default: 0)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Start execution from a specific step number (e.g. 13) or filename (e.g. step_13_lr.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scheduled execution plan without running the scripts",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PD-MCI Nomogram: Pipeline Runner")
    print("Sequence: Strict numerical step order (step_01 to step_16)")
    print("=" * 60)

    missing_scripts = [
        script for _, script in PIPELINE_MANIFEST if not os.path.isfile(script)
    ]
    if missing_scripts:
        print("\n[FAIL] Missing required pipeline script(s):")
        for script in missing_scripts:
            print(f"  - {script}")
        sys.exit(1)

    discovered = set(
        glob.glob("step_*.py") + glob.glob("step_*.R") + glob.glob("step_*.r")
    )
    ignored = sorted(discovered - {script for _, script in PIPELINE_MANIFEST})
    if ignored:
        print("\n[Manifest] Ignoring step script(s) not in the formal pipeline:")
        for script in ignored:
            print(f"  - {script}")

    rscript_path = get_rscript_path()

    print(f"\n[Manifest] Formal pipeline contains {len(PIPELINE_MANIFEST)} script(s):")
    for i, (script_type, script) in enumerate(PIPELINE_MANIFEST, 1):
        print(f"  [{script_type:6s}] {i:2d}. {script}")
    print(f"  Rscript executable: {rscript_path}")

    execution_queue = []
    for ftype, f in PIPELINE_MANIFEST:
        if args.r_only and ftype != "R":
            continue
        if args.python_only and ftype != "Python":
            continue
        if ftype == "Python":
            execution_queue.append(
                {
                    "type": "Python",
                    "script": f,
                    "runner": sys.executable,
                    "extra_args": None,
                }
            )
        else:
            execution_queue.append(
                {
                    "type": "R",
                    "script": f,
                    "runner": rscript_path,
                    "extra_args": None,
                }
            )

    if not execution_queue:
        print("\nNo scripts matched the execution criteria.")
        return

    # Apply the requested starting step or skip count.
    if args.start_from:
        start_target = args.start_from.strip().lower()
        start_idx = None
        for idx, item in enumerate(execution_queue):
            base = os.path.basename(item["script"]).lower()
            num_match = re.search(r"step_(\d+)", base)
            if start_target in base:
                start_idx = idx
                break
            if num_match and num_match.group(1).lstrip("0") == start_target.lstrip("0"):
                start_idx = idx
                break
        if start_idx is not None:
            execution_queue = execution_queue[start_idx:]
            print(
                f"\nStarting from: {execution_queue[0]['script']} (index {start_idx + 1})"
            )
        else:
            print(
                f"\n[Warning] Could not find script matching '--start-from {args.start_from}'. Running from start."
            )

    if args.skip > 0:
        if args.skip < len(execution_queue):
            print(f"\nSkipping first {args.skip} script(s).")
            execution_queue = execution_queue[args.skip :]
        else:
            print(
                f"\n[Warning] Skip count ({args.skip}) >= total scripts ({len(execution_queue)}). Nothing to run."
            )
            return

    total_count = len(execution_queue)
    if args.dry_run:
        print("\n" + "=" * 60)
        print(f"[DRY RUN] Planned Execution Queue ({total_count} script(s)):")
        print("=" * 60)
        for idx, item in enumerate(execution_queue, 1):
            cmd = [item["runner"], item["script"]]
            if item["extra_args"]:
                cmd.extend(item["extra_args"])
            print(f"  [{idx:2d}/{total_count}] [{item['type']:6s}] {' '.join(cmd)}")
        print("\n[DRY RUN] Completed without running any scripts.")
        return

    print("\n" + "=" * 60)
    print(f"Beginning execution of {total_count} script(s)")
    print("=" * 60)

    overall_start = time.time()
    successful = []
    failed = []

    for idx, item in enumerate(execution_queue, 1):
        script = item["script"]
        stype = item["type"]
        runner = item["runner"]
        extra = item["extra_args"]

        print(f"\n[{idx}/{total_count}] [{stype}] Executing: {script}")
        ok = run_script(script, runner, extra)
        if ok:
            successful.append(script)
        else:
            failed.append(script)
            print(f"\n[ABORT] Pipeline stopped due to failure in {script}")
            break

    total_time = time.time() - overall_start

    print("\n" + "=" * 60)
    print("Pipeline Execution Summary")
    print("=" * 60)
    print(f"Total executed : {len(successful) + len(failed)}/{total_count}")
    print(f"Successful     : {len(successful)}")
    print(f"Failed         : {len(failed)}")
    print(f"Total elapsed  : {total_time:.2f}s")

    if failed:
        print(f"\nFailed script: {failed[0]}")
        sys.exit(1)
    else:
        print("\nAll executed scripts completed successfully!")


if __name__ == "__main__":
    main()
