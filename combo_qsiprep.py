#!/usr/bin/env python3
"""Interactive QSIPrep 1.0.0 launcher for the COMBO cohorts."""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def get_system_specs():
    cores = os.cpu_count() or 4
    total_ram_gb = 8.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    total_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
    except OSError:
        pass
    return cores, total_ram_gb, 1.48


SYSTEM_CORES, SYSTEM_RAM_GB, EC2_HOURLY_RATE = get_system_specs()

CORES_PER_CONTAINER = 8
MIN_GB_PER_CONTAINER = 8
RESERVED_RAM_GB = max(8.0, SYSTEM_RAM_GB * 0.10)
INSTANCE_PRESETS = {(16, 62): (2, 8, 22.0)}

BASE_BIDS_DIR = Path(
    "/Marshlab/Projects/COMBO_8105/5_Data/marscombo_scan_data/marscombo/rawdata"
)
OUTPUT_QSIPREP_MOMS_DIR = Path(
    "/Marshlab/Projects/COMBO_8105/5_Data/marscombo_scan_data/marscombo/derivatives/qsiprep1.0.0"
)
OUTPUT_QSIPREP_BABIES_DIR = Path(
    "/MRI_Projects/marscombo/derivatives/qsiprep1.0.0"
)
GLOBAL_WORK_DIR = Path("/Marshlab/data/proc_work_dirs/qsiprep_wd")
FS_LICENSE = Path("/var/tmp/freesurfer_lic/license.txt")

MOM_QSIPREP_PARAMS = [
    "--output-resolution",
    "1.71",
    "--separate-all-dwis",
    "--dwi-only",
]
BABY_QSIPREP_PARAMS = [
    "--output-resolution",
    "1.71",
    "--subject-anatomical-reference",
    "sessionwise",
    "--anat-modality",
    "T2w",
    "--infant",
    "--unringing-method",
    "rpg",
    "--denoise-method",
    "dwidenoise",
    "--b1-biascorrect-stage",
    "none",
    "--pepolar-method",
    "TOPUP+DRBUDDI",
]


class Subject:
    def __init__(self, path):
        self.path = Path(path)
        self.raw_name = self.path.name
        self.bids_id = (
            self.raw_name.replace("sub-", "", 1)
            if self.raw_name.startswith("sub-")
            else self.raw_name
        )
        self.name = f"sub-{self.bids_id}"
        self.has_bids = False
        self.has_raw_dwi = False
        self.has_complete_dwi = False
        self.has_output = False
        self.dwi_runs = []
        self.incomplete_dwi_runs = []

    def scan(self, output_qsiprep_dir):
        self.has_bids = self.path.exists()
        self.dwi_runs, self.incomplete_dwi_runs = find_dwi_runs(self.path)
        self.has_raw_dwi = bool(self.dwi_runs or self.incomplete_dwi_runs)
        self.has_complete_dwi = bool(self.dwi_runs)
        self.has_output = (output_qsiprep_dir / self.name).is_dir()


def find_dwi_runs(subject_dir):
    """Return complete and incomplete DWI run stems beneath a BIDS subject."""
    dwi_dir_files = {}
    for root, _, filenames in os.walk(subject_dir):
        if Path(root).name != "dwi":
            continue
        for filename in filenames:
            if not filename.endswith(
                ("_dwi.nii", "_dwi.nii.gz", "_dwi.bval", "_dwi.bvec", "_dwi.json")
            ):
                continue
            path = Path(root) / filename
            stem = str(path)
            for suffix in ("_dwi.nii.gz", "_dwi.nii", "_dwi.bval", "_dwi.bvec", "_dwi.json"):
                if stem.endswith(suffix):
                    run_stem = stem[: -len(suffix)]
                    break
            extensions = dwi_dir_files.setdefault(run_stem, set())
            extensions.add(path.suffix)
            if filename.endswith("_dwi.nii.gz"):
                extensions.add(".nii.gz")

    complete = []
    incomplete = []
    for run_stem, extensions in sorted(dwi_dir_files.items()):
        if ({".nii", ".nii.gz"} & extensions) and {".bval", ".bvec"} <= extensions:
            complete.append(run_stem)
        else:
            incomplete.append(run_stem)
    return complete, incomplete


def output_dir_for(cohort_target):
    if cohort_target == "moms":
        return OUTPUT_QSIPREP_MOMS_DIR
    if cohort_target == "babies":
        return OUTPUT_QSIPREP_BABIES_DIR
    raise ValueError(f"Unknown cohort: {cohort_target}")


def get_lustre_mds_usage():
    try:
        result = subprocess.run(
            ["lfs", "df"], capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0 and len(result.stdout.strip().splitlines()) > 1:
            return result.stdout.strip().splitlines()[1].split()[4]
    except (OSError, subprocess.SubprocessError):
        pass
    return "N/A"


def execute_container_and_profile(cmd_list, log_path):
    time_cmd = ["/usr/bin/time", "-v"] + cmd_list
    cmd_str = " ".join(cmd_list)
    start_time = time.time()
    peak_lustre = "N/A"
    min_lustre = get_lustre_mds_usage()

    with open(log_path, "w") as log_file:
        log_file.write(f"=== LAUNCHING QSIPREP ===\nCommand: {cmd_str}\n\n")
        log_file.flush()
        try:
            process = subprocess.Popen(
                time_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            assert process.stdout is not None
            for output in process.stdout:
                log_file.write(output)
                log_file.flush()
            return_code = process.wait()
            duration = time.time() - start_time
            max_mem_gb = 0.0
            try:
                with open(log_path, "r") as completed_log:
                    for line in completed_log:
                        if "Maximum resident set size" in line:
                            max_mem_gb = int(line.split(":")[-1].strip()) / (
                                1024 * 1024
                            )
                            break
            except (OSError, ValueError):
                pass
            return (
                return_code == 0,
                duration,
                max_mem_gb,
                min_lustre,
                peak_lustre,
                cmd_str,
                "",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return (
                False,
                time.time() - start_time,
                0.0,
                min_lustre,
                peak_lustre,
                cmd_str,
                str(exc),
            )


def belongs_to_cohort(subject_id, cohort_target):
    if cohort_target == "moms":
        return subject_id.lower().endswith("p")
    if cohort_target == "babies":
        return bool(subject_id) and subject_id[-1].isdigit()
    raise ValueError(f"Unknown cohort: {cohort_target}")


def get_subjects(cohort_target):
    subjects = []
    if not BASE_BIDS_DIR.exists():
        return subjects

    output_qsiprep_dir = output_dir_for(cohort_target)
    for item in sorted(BASE_BIDS_DIR.iterdir()):
        if not item.is_dir() or not item.name.startswith("sub-"):
            continue
        subject = Subject(item)
        if not belongs_to_cohort(subject.bids_id, cohort_target):
            continue
        subject.scan(output_qsiprep_dir)
        subjects.append(subject)
    return subjects


def run_audit(subjects, cohort_target):
    ready, completed, missing = [], [], []
    for subject in subjects:
        if subject.has_output:
            completed.append(subject)
        elif subject.has_complete_dwi:
            ready.append(subject)
        else:
            missing.append(subject)

    print(f"\n====== QSIPREP AUDIT ({cohort_target.upper()}) ======")
    print(
        f"READY: {len(ready)} | Completed: {len(completed)} | "
        f"Missing Inputs: {len(missing)}\n"
    )
    if ready:
        ready.sort(key=lambda subject: subject.name)
        print("READY subjects:")
        for subject in ready:
            print(f"  {subject.name} ({len(subject.dwi_runs)} complete DWI runs)")
        print()
    if missing:
        print("MISSING INPUTS subjects:")
        for subject in sorted(missing, key=lambda item: item.name):
            if subject.incomplete_dwi_runs:
                detail = f"{len(subject.incomplete_dwi_runs)} incomplete DWI runs"
            else:
                detail = "no DWI files found"
            print(f"  {subject.name} ({detail})")
        print()
    return ready


def qsiprep_params_for(cohort_target):
    if cohort_target == "moms":
        return MOM_QSIPREP_PARAMS
    if cohort_target == "babies":
        return BABY_QSIPREP_PARAMS
    raise ValueError(f"Unknown cohort: {cohort_target}")


def build_docker_command(subject, nthreads, mem_mb, cohort_target):
    sub_work_dir = GLOBAL_WORK_DIR / subject.name
    output_dir = output_dir_for(cohort_target)
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{BASE_BIDS_DIR}:/raw_bids:ro",
        "-v",
        f"{output_dir}:/out",
        "-v",
        f"{sub_work_dir}:/work",
        "-v",
        f"{FS_LICENSE}:/fs_license.txt:ro",
        "pennlinc/qsiprep:1.0.0",
        "/raw_bids",
        "/out",
        "participant",
        "--participant-label",
        subject.bids_id,
        "--fs-license-file",
        "/fs_license.txt",
        "--work-dir",
        "/work",
        "--bids-database-dir",
        "/work/bids_db",
        "--nthreads",
        str(nthreads),
        "--omp-nthreads",
        str(nthreads),
        "--mem-mb",
        str(mem_mb),
        "--notrack",
        *qsiprep_params_for(cohort_target),
        "-v",
    ]


def process_subject_wrapper(subject, nthreads, mem_mb, cohort_target):
    sub_work_dir = GLOBAL_WORK_DIR / subject.name
    sub_work_dir.mkdir(parents=True, exist_ok=True)
    output_dir_for(cohort_target).mkdir(parents=True, exist_ok=True)
    docker_cmd = build_docker_command(subject, nthreads, mem_mb, cohort_target)

    print(f"[{subject.name}] Executing preprocessing...")
    success, duration, max_mem, min_lust, peak_lust, cmd_str, error = (
        execute_container_and_profile(docker_cmd, f"{subject.name}_qsiprep.log")
    )
    if success and sub_work_dir.exists():
        shutil.rmtree(sub_work_dir)
    return {
        "subject": subject.name,
        "status": "SUCCESS" if success else "FAILED",
        "duration_min": duration / 60,
        "error": error,
    }


def default_resource_plan(n_subjects):
    usable_gb = max(0.0, SYSTEM_RAM_GB - RESERVED_RAM_GB)
    preset = INSTANCE_PRESETS.get((SYSTEM_CORES, round(SYSTEM_RAM_GB)))
    if preset:
        workers, nthreads, mem_gb = preset
        return max(1, min(workers, n_subjects)), nthreads, mem_gb
    workers = max(
        1,
        min(
            SYSTEM_CORES // CORES_PER_CONTAINER,
            int(usable_gb // MIN_GB_PER_CONTAINER),
            n_subjects,
        ),
    )
    nthreads = max(1, SYSTEM_CORES // workers)
    mem_gb = round(usable_gb / workers, 1)
    return workers, nthreads, mem_gb


def ask_number(label, default, cast, minimum):
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"  Enter a number >= {minimum}.")


def prompt_resource_plan(n_subjects):
    d_workers, d_threads, d_mem = default_resource_plan(n_subjects)
    usable_gb = max(0.0, SYSTEM_RAM_GB - RESERVED_RAM_GB)
    print(
        f"\nDetected: {SYSTEM_CORES} CPUs | {SYSTEM_RAM_GB:.0f} GB RAM "
        f"(reserving {RESERVED_RAM_GB:.0f} GB for the system, "
        f"{usable_gb:.0f} GB usable)"
    )
    print(
        f"Default plan: {d_workers} concurrent containers x "
        f"{d_threads} threads x {d_mem} GB RAM"
    )
    if input("Customize? (y/N): ").strip().lower() != "y":
        return d_workers, d_threads, d_mem

    while True:
        workers = ask_number("Concurrent containers", d_workers, int, 1)
        threads = ask_number(
            "Threads per container", max(1, SYSTEM_CORES // workers), int, 1
        )
        mem_gb = ask_number(
            "RAM per container (GB)",
            round(usable_gb / workers, 1),
            float,
            MIN_GB_PER_CONTAINER,
        )
        if workers * mem_gb > usable_gb:
            print(
                f"  {workers} x {mem_gb} GB = {workers * mem_gb:.0f} GB exceeds "
                f"the {usable_gb:.0f} GB usable (must leave "
                f"{RESERVED_RAM_GB:.0f} GB for the system). Try again."
            )
            continue
        if workers * threads > SYSTEM_CORES:
            print(
                f"  Warning: {workers * threads} total threads oversubscribes "
                f"{SYSTEM_CORES} CPUs."
            )
        return workers, threads, mem_gb


def pipeline_execution_loop(ready_subjects, cohort_target):
    if not ready_subjects:
        return
    workers, nthreads, mem_gb = prompt_resource_plan(len(ready_subjects))
    mem_mb = int(mem_gb * 1024)
    confirmation = input(
        f"Launch {workers} concurrent containers "
        f"({nthreads} threads | {mem_mb} MB RAM each)? (y/n): "
    )
    if confirmation.lower() != "y":
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_subject_wrapper,
                subject,
                nthreads,
                mem_mb,
                cohort_target,
            ): subject
            for subject in ready_subjects
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            detail = f" | Error: {result['error']}" if result["error"] else ""
            print(
                f"[{result['status']}] {result['subject']} | "
                f"Time: {result['duration_min']:.2f} min{detail}"
            )

    check_dwi_outputs(ready_subjects, cohort_target)


def check_dwi_outputs(subjects, cohort_target):
    """Report subjects for which QSIPrep did not create a DWI output folder."""
    output_dir = output_dir_for(cohort_target)
    missing = []
    for subject in subjects:
        subject_output = output_dir / subject.name
        dwi_dirs = (
            list(subject_output.rglob("dwi"))
            if subject_output.is_dir()
            else []
        )
        if not dwi_dirs:
            missing.append(subject.name)
    print("\n====== OUTPUT CHECK (<subject>/.../dwi) ======")
    if missing:
        print(f"MISSING DWI output for {len(missing)} of {len(subjects)} subjects run:")
        for subject in missing:
            print(f"  - {subject}")
    else:
        print(f"All {len(subjects)} subjects have a DWI output folder.")


def delete_failed_subjects(cohort_target):
    """Delete selected QSIPrep output and work directories after confirmation."""
    raw_ids = input(
        "Subject IDs to delete (space-separated, e.g. 1755138P sub-R1762041): "
    ).split()
    names = []
    for raw_id in raw_ids:
        name = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        if len(name) <= 4 or "/" in name or ".." in name:
            print(f"  Skipping invalid ID: {raw_id}")
        elif name not in names:
            names.append(name)

    roots = [output_dir_for(cohort_target), GLOBAL_WORK_DIR]
    targets = [
        root / name
        for name in names
        for root in roots
        if (root / name).is_dir()
    ]
    for name in names:
        if not any(target.name == name for target in targets):
            print(f"  Nothing found for {name}")
    if not targets:
        return False

    print("\nThe following folders will be PERMANENTLY deleted:")
    for target in targets:
        print(f"  {target}")
    if input(f"Delete these {len(targets)} folders? (y/n): ").strip().lower() != "y":
        print("Cancelled.")
        return False
    for target in targets:
        try:
            shutil.rmtree(target)
            print(f"  Deleted {target}")
        except OSError as exc:
            print(f"  [ERROR] Could not delete {target}: {exc}")
    return True


def main():
    print("=== QSIPREP INITIALIZATION ===")
    cohort_target = (
        "moms" if input("Cohort [1=Moms, 2=Babies]: ").strip() == "1" else "babies"
    )
    subjects = get_subjects(cohort_target)

    while True:
        print(
            "\n1. Audit | 2. Run All | 3. Run TEST (First 3) | "
            "4. Delete failed subjects | 0. Exit"
        )
        choice = input("Select Task: ")
        if choice == "0":
            return 0
        if choice == "1":
            run_audit(subjects, cohort_target)
        elif choice == "2":
            pipeline_execution_loop(run_audit(subjects, cohort_target), cohort_target)
        elif choice == "3":
            pipeline_execution_loop(
                run_audit(subjects, cohort_target)[:3], cohort_target
            )
        elif choice == "4":
            delete_failed_subjects(cohort_target)
        if choice in ("2", "3", "4"):
            subjects = get_subjects(cohort_target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
