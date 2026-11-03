import os
import sys
import shutil
import subprocess
import concurrent.futures
import time
import json
from pathlib import Path

def get_system_specs():
    cores = os.cpu_count() or 4
    total_ram_gb = 8.0
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    total_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
    except: pass
    return cores, total_ram_gb, 1.48

SYSTEM_CORES, SYSTEM_RAM_GB, EC2_HOURLY_RATE = get_system_specs()

#BASE_BIDS_DIR = Path("/MRI_Projects/marscombo/rawdata")
BASE_BIDS_DIR = Path("/Marshlab/Projects/COMBO_8105/5_Data/marscombo_scan_data/marscombo/rawdata")
# UPDATED PATH: Pointing to the writable drive for Moms
# INPUT_QSIPREP_MOMS_DIR = Path("/MRI_Projects/marscombo/derivatives/qsiprep1.0.0")
INPUT_QSIPREP_MOMS_DIR = Path("/Marshlab/Projects/COMBO_8105/5_Data/marscombo_scan_data/marscombo/derivatives/qsiprep1.0.0")
INPUT_QSIPREP_BABIES_DIR = Path("/MRI_Projects/marscombo/derivatives/qsiprep1.0.0")
OUTPUT_DERIVATIVES_DIR = Path("/Marshlab/data/COMBO/derivatives/qsirecon1.0.0")
GLOBAL_WORK_DIR = Path("/Marshlab/data/proc_work_dirs/qsirecon_wd")
FS_LICENSE = Path("/var/tmp/freesurfer_lic/license.txt")

class Subject:
    def __init__(self, path):
        self.path = Path(path)
        self.raw_name = self.path.name  
        self.bids_id = self.raw_name.replace("sub-", "", 1) if self.raw_name.startswith("sub-") else self.raw_name
        self.name = f"sub-{self.bids_id}"
        self.has_bids = False
        self.has_qsiprep_input = False
        self.has_output = False
        
    def scan(self, input_qsiprep_dir):
        if self.path.exists(): self.has_bids = True
        if (input_qsiprep_dir / self.name).exists(): self.has_qsiprep_input = True
        if (OUTPUT_DERIVATIVES_DIR / self.name).exists(): self.has_output = True

def ensure_dataset_description(target_dir):
    """
    Creates a perfect dataset_description.json in the target directory if it is missing.
    Includes the critical DatasetLinks dictionary required by QSIRecon 1.0.0.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    desc_path = target_dir / "dataset_description.json"
    
    if not desc_path.exists():
        print(f"\n[SYSTEM] Generating missing dataset_description.json with DatasetLinks at {desc_path}...")
        try:
            with open(desc_path, "w") as f:
                json.dump({
                    "Name": "QSIPrep Outputs",
                    "BIDSVersion": "1.4.0",
                    "DatasetType": "derivative",
                    "GeneratedBy": [{"Name": "qsiprep"}],
                    "DatasetLinks": {"raw": "/raw_bids"}
                }, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to write dataset description: {e}")

def get_lustre_mds_usage():
    try:
        res = subprocess.run(["lfs", "df"], capture_output=True, text=True, timeout=1)
        if res.returncode == 0 and len(res.stdout.strip().split('\n')) > 1:
            return res.stdout.strip().split('\n')[1].split()[4]
    except: pass
    return "N/A"

def execute_container_and_profile(cmd_list, log_path):
    time_cmd = ["/usr/bin/time", "-v"] + cmd_list
    cmd_str = " ".join(cmd_list)
    start_time = time.time()
    peak_lustre, min_lustre = "N/A", get_lustre_mds_usage()
    
    with open(log_path, "w") as log_file:
        log_file.write(f"=== LAUNCHING QSIRECON ===\nCommand: {cmd_str}\n\n")
        log_file.flush()
        try:
            process = subprocess.Popen(time_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None: break
                if output:
                    log_file.write(output)
                    log_file.flush()
            process.wait()
            duration = time.time() - start_time
            max_mem_gb = 0.0
            try:
                with open(log_path, "r") as lf:
                    for line in lf:
                        if "Maximum resident set size" in line:
                            max_mem_gb = int(line.split(':')[-1].strip()) / (1024 * 1024)
                            break
            except: pass
            return (process.returncode == 0), duration, max_mem_gb, min_lustre, peak_lustre, cmd_str, ""
        except Exception as e:
            return False, time.time() - start_time, 0.0, min_lustre, peak_lustre, cmd_str, str(e)

def get_subjects(cohort_target):
    subjects = []
    if not BASE_BIDS_DIR.exists(): return subjects
    input_qsiprep_dir = INPUT_QSIPREP_MOMS_DIR if cohort_target == "moms" else INPUT_QSIPREP_BABIES_DIR
    
    # Auto-generate the JSON before scanning starts
    ensure_dataset_description(input_qsiprep_dir)
    
    for item in BASE_BIDS_DIR.iterdir():
        if item.is_dir():
            sub = Subject(item)
            if cohort_target == "moms" and not sub.bids_id.lower().endswith('p'): continue
            if cohort_target == "babies" and not sub.bids_id[-1].isdigit(): continue
            sub.scan(input_qsiprep_dir)
            subjects.append(sub)
    return subjects

def run_audit(subs, cohort_target):
    ready, completed, missing = [], [], []
    for s in subs:
        if not s.has_qsiprep_input: missing.append(s)
        elif s.has_output: completed.append(s)
        else: ready.append(s)
    print(f"\n====== QSIRECON AUDIT ({cohort_target.upper()}) ======")
    print(f"READY: {len(ready)} | Completed: {len(completed)} | Missing Inputs: {len(missing)}\n")
    return ready

def process_subject_wrapper(s, nthreads, mem_mb, cohort_target):
    sub_work_dir = GLOBAL_WORK_DIR / s.name
    sub_work_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    
    input_qsiprep_dir = INPUT_QSIPREP_MOMS_DIR if cohort_target == "moms" else INPUT_QSIPREP_BABIES_DIR
    
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{BASE_BIDS_DIR}:/raw_bids:ro",
        "-v", f"{input_qsiprep_dir}:/qsiprep_input:ro",
        "-v", f"{OUTPUT_DERIVATIVES_DIR}:/out",
        "-v", f"{sub_work_dir}:/work",
        "-v", f"{FS_LICENSE}:/fs_license.txt:ro",
        "pennlinc/qsirecon:1.0.0",  
        "/qsiprep_input", "/out",  
        "participant",
        "--input-type", "qsiprep", 
        "--participant-label", s.bids_id,               
        "--fs-license-file", "/fs_license.txt",
        "--work-dir", "/work",
        "--bids-database-dir", "/work/clean_sbids_db",
        "--nthreads", str(nthreads),
        "--mem", str(mem_mb),
        "--notrack"
    ]
    
    if cohort_target == "babies":
        docker_cmd.extend(["--recon-spec", "hbcd_scalar_maps", "--output-resolution", "1.7", "--report-output-level", "session", "--infant", "-v"])
    else:
        docker_cmd.extend(["--recon-spec", "hbcd_scalar_maps", "--output-resolution", "1.7", "--report-output-level", "session", "-v"])
        
    print(f"[{s.name}] Executing reconstruction...")
    success, dur, max_mem, min_lust, peak_lust, cmd_str, err_msg = execute_container_and_profile(docker_cmd, f"{s.name}_qsirecon.log")
    if success and sub_work_dir.exists(): shutil.rmtree(sub_work_dir)
    return {"subject": s.name, "status": "SUCCESS" if success else "FAILED", "duration_min": dur / 60, "error": err_msg}

def pipeline_execution_loop(ready_subs, cohort_target):
    if not ready_subs: return
    workers = max(1, min(2, SYSTEM_CORES // 4)) 
    nthreads = max(2, SYSTEM_CORES // workers)
    mem_mb = int(max(4.0, (SYSTEM_RAM_GB * 0.85) / workers) * 1024)
    
    confirm = input(f"Launch {workers} concurrent containers ({nthreads} threads | {mem_mb} MB RAM each)? (y/n): ")
    if confirm.lower() != 'y': return
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_subject_wrapper, s, nthreads, mem_mb, cohort_target): s for s in ready_subs}
        for future in concurrent.futures.as_completed(futures):
            m = future.result()
            print(f"[{m['status']}] {m['subject']} | Time: {m['duration_min']:.2f} min")

def main():
    print("=== QSIRECON INITIALIZATION ===")
    cohort_target = "moms" if input("Cohort [1=Moms, 2=Babies]: ").strip() == "1" else "babies"
    subs = get_subjects(cohort_target)
    
    while True:
        print("\n1. Audit | 2. Run All | 3. Run TEST (First 3) | 0. Exit")
        choice = input("Select Task: ")
        if choice == "0": sys.exit(0)
        elif choice == "1": run_audit(subs, cohort_target)
        elif choice == "2": pipeline_execution_loop(run_audit(subs, cohort_target), cohort_target)
        elif choice == "3": pipeline_execution_loop(run_audit(subs, cohort_target)[:3], cohort_target)

if __name__ == "__main__":
    main()
