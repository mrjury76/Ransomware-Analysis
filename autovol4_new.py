import argparse
import subprocess
import csv
import io
import os
from datetime import datetime

# ===== MULTI-FAMILY CONFIG =====
# Maps family name -> list of process name substrings to match (case-insensitive)
FAMILY_PROCESS_NAMES = {
    "WannaCry":   ["wannacry", "tasksche", "mssecsvc"],
    "Cerber":     ["cerber"],
    "Jigsaw":     ["jigsaw"],
    "Dharma":     ["dharma"],
    "Benign":     [],  # no PID filter — capture all processes as baseline
}

# Plugins that are NOT filtered by PID (no PID column or global scope)
NO_PID_FILTER_PLUGINS = {"windows.malfind", "windows.filescan", "windows.svcscan", "windows.netstat"}

VOL_COMMAND = "/home/patrick/tools/volatility3/venv/bin/vol"
PLUGINS = [
    "windows.pslist",       # process list - process spawning over time
    "windows.psscan",       # catches hidden processes pslist misses
    "windows.cmdline",      # command line args per process
    "windows.dlllist",      # loaded DLLs per process
    "windows.ldrmodules",   # hidden/injected DLL detection
    "windows.malfind",      # injected executable memory regions
    "windows.vadinfo",      # virtual address descriptors
    "windows.handles",      # files/registry/objects opened
    "windows.filescan",     # file objects in memory
    "windows.svcscan",      # service installation
    "windows.privileges",   # privilege escalation tracking
    "windows.netstat",      # network connections
]
# ================================

def prompt_family():
    print("")
    print("Available families:")
    for k in sorted(FAMILY_PROCESS_NAMES.keys()):
        print(f"  - {k}")
    print("")
    family = input("Which family? ").strip()
    if family not in FAMILY_PROCESS_NAMES:
        print(f"[!] Unknown family '{family}'. Add it to FAMILY_PROCESS_NAMES or check spelling.")
        exit(1)
    return family

def resolve_memory_image(family, version):
    """
    Try two locations for the vmem:
      1. Old sequential format: C:/VMs/Windows 10 x64-Snapshot{N}.vmem
      2. New ps1 format:        D:/Patrick/VMSnapshots/{family}_*/run*/*/{snapname}.vmem
    Returns the first path that exists, or None.
    """
    # Old format (numbered snapshots on C:)
    old_path = (f"/mnt/c/Users/Patrick/Documents/Virtual Machines/Windows 10 x64/"
                f"Windows 10 x64-Snapshot{version}.vmem")
    if os.path.exists(old_path):
        return old_path

    # New format: walk D: for any vmem whose filename contains the version token
    new_base = "/mnt/d/Patrick/VMSnapshots"
    if os.path.isdir(new_base):
        for root, dirs, files in os.walk(new_base):
            for fname in files:
                if fname.endswith(".vmem") and version in fname:
                    return os.path.join(root, fname)

    return None

def get_malware_pids(family, memory_image, log_fn):
    """Run pstree, return set of PIDs belonging to the target family process tree.
    For Benign, returns empty set immediately (no PID filter needed)."""
    name_hints = FAMILY_PROCESS_NAMES[family]

    if not name_hints:
        log_fn(f"\n[+] {family} family — no PID filter, capturing all processes.")
        return set()

    log_fn(f"\n[+] Running windows.pstree to find {family} PIDs (hints: {name_hints})...")

    result = subprocess.run(
        [VOL_COMMAND, "-q", "-r", "csv", "-f", memory_image, "windows.pstree"],
        capture_output=True, text=True
    )

    processes = []
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        try:
            pid  = str(int(row.get("PID",  "").strip()))
            ppid = str(int(row.get("PPID", "").strip()))
            name = row.get("ImageFileName", "").strip()
            processes.append({"pid": pid, "ppid": ppid, "name": name})
        except ValueError:
            continue

    log_fn(f"    -> Parsed {len(processes)} processes")

    pid_set = set()
    for p in processes:
        if any(hint in p["name"].lower() for hint in name_hints):
            pid_set.add(p["pid"])
            log_fn(f"    [+] Found {family} process: {p['name']} PID={p['pid']}")

    if not pid_set:
        log_fn(f"    [!] No {family} processes found. Running without PID filter.")
        return set()

    # Collect all descendants
    added = True
    while added:
        added = False
        for p in processes:
            if p["ppid"] in pid_set and p["pid"] not in pid_set:
                pid_set.add(p["pid"])
                log_fn(f"    [+] Added child: {p['name']} PID={p['pid']}")
                added = True

    log_fn(f"    [+] Total {family} PIDs: {sorted(pid_set, key=int)}")
    return pid_set

def filter_csv_by_pid(raw_csv, pid_set):
    reader = csv.DictReader(io.StringIO(raw_csv))
    rows = []
    for row in reader:
        for key, val in row.items():
            if key.upper() == "PID":
                try:
                    if str(int(val.strip())) in pid_set:
                        rows.append(row)
                except (ValueError, AttributeError):
                    pass
                break
    return rows, reader.fieldnames

def run_analysis(family, memory_image, output_dir, executor=None):
    """Core analysis logic. Called from both CLI and interactive modes.

    If executor is passed, plugin tasks are submitted to the shared pool.
    Otherwise a local pool is created (single-snapshot mode).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    csv_out  = os.path.join(output_dir, "vol3_combined.csv")
    log_file = os.path.join(output_dir, "run.log")
    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    start = datetime.now()
    log("=" * 50)
    log(f" Volatility 3 Feature Extraction")
    log(f" Family:    {family}")
    log(f" Snapshot:  {memory_image}")
    log(f" Started:   {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 50)

    malware_pids   = get_malware_pids(family, memory_image, log)
    all_rows       = []
    all_fieldnames = []
    seen_fields    = set()

    def run_plugin(plugin):
        plugin_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = subprocess.run(
            [VOL_COMMAND, "-q", "-r", "csv", "-f", memory_image, plugin],
            capture_output=True, text=True
        )
        return plugin, plugin_ts, result.stdout

    # Submit all plugins to the shared pool, or create a local one
    owns_executor = executor is None
    if owns_executor:
        executor = ThreadPoolExecutor(max_workers=12)

    futures = {executor.submit(run_plugin, p): p for p in PLUGINS}
    plugin_results = {}
    for future in as_completed(futures):
        plugin, plugin_ts, raw = future.result()
        log(f"[+] Done: {plugin}")
        plugin_results[plugin] = (plugin_ts, raw)

    if owns_executor:
        executor.shutdown(wait=False)

    # Merge results in original plugin order so combined CSV is consistent
    for plugin in PLUGINS:
        if plugin not in plugin_results:
            continue
        plugin_ts, raw = plugin_results[plugin]

        if not raw.strip():
            log(f"    -> {plugin}: no output")
            continue

        raw_path = os.path.join(output_dir, f"{plugin}.csv")
        with open(raw_path, "w") as f:
            f.write(raw)

        if malware_pids and plugin not in NO_PID_FILTER_PLUGINS:
            rows, fieldnames = filter_csv_by_pid(raw, malware_pids)
        else:
            reader = csv.DictReader(io.StringIO(raw))
            rows = list(reader)
            fieldnames = reader.fieldnames

        log(f"    {plugin}: {len(rows)} rows")

        for row in rows:
            row["family"]       = family
            row["plugin"]       = plugin
            row["timestamp"]    = plugin_ts
            row["memory_image"] = memory_image

        for fn in (fieldnames or []):
            if fn not in seen_fields:
                all_fieldnames.append(fn)
                seen_fields.add(fn)

        all_rows.extend(rows)

    if not all_rows:
        log("\n[-] No data collected.")
        return

    priority = ["family", "plugin", "timestamp", "PID", "PPID", "ImageFileName", "memory_image"]
    final_fields = []
    seen = set()
    for k in priority:
        if k not in seen:
            final_fields.append(k)
            seen.add(k)
    for k in all_fieldnames:
        if k not in seen:
            final_fields.append(k)
            seen.add(k)
    for row in all_rows:
        for k in row.keys():
            if k not in seen:
                final_fields.append(k)
                seen.add(k)

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    log(f"\n[✓] CSV saved: {csv_out}")
    log(f"[✓] Total rows: {len(all_rows)}")
    log(f"[+] Done. Total time: {datetime.now() - start}")
    log(f"[+] Output directory: {output_dir}")


def batch_mode(scan_dir, only_families=None):
    """
    Walk scan_dir, find every .vmem file, read the family from meta.json
    in the same folder, and run analysis into that same folder.
    Skips any folder that already has vol3_combined.csv.

    If only_families is a set/list, only process snapshots whose family is in
    that set (e.g. {"WannaCry", "Benign"}).

    Uses a single shared thread pool for all plugin work across all snapshots.
    12 workers = 12 cores always busy regardless of how many snapshots remain.
    """
    import json
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor

    MAX_WORKERS = multiprocessing.cpu_count()

    vmem_files = []
    for root, _, files in os.walk(scan_dir):
        for fname in files:
            if fname.endswith(".vmem"):
                vmem_files.append(os.path.join(root, fname))

    if not vmem_files:
        print(f"[-] No .vmem files found under {scan_dir}")
        return

    print(f"\n[+] Found {len(vmem_files)} vmem file(s) to process")
    print(f"[+] Shared pool workers: {MAX_WORKERS}")

    # Build work list
    work = []
    skipped = 0
    failed  = 0

    for vmem_path in sorted(vmem_files):
        snap_dir   = os.path.dirname(vmem_path)
        csv_out    = os.path.join(snap_dir, "vol3_combined.csv")
        meta_path  = os.path.join(snap_dir, "meta.json")

        if os.path.exists(csv_out):
            print(f"[~] Skipping (already done): {vmem_path}")
            skipped += 1
            continue

        family = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    family = json.load(f).get("family")
            except Exception:
                pass

        if not family:
            parts = vmem_path.split(os.sep)
            for part in parts:
                for known in FAMILY_PROCESS_NAMES:
                    if part.startswith(known):
                        family = known
                        break
                if family:
                    break

        if not family or family not in FAMILY_PROCESS_NAMES:
            print(f"[!] Cannot determine family for {vmem_path} — skipping")
            failed += 1
            continue

        if only_families and family not in only_families:
            skipped += 1
            continue

        work.append((family, vmem_path, snap_dir))

    if not work:
        print(f"\n[✓] Batch complete — done: 0, skipped: {skipped}, failed: {failed}")
        return

    print(f"[+] Queued {len(work)} snapshot(s) for processing")

    # One shared pool — all plugin tasks from all snapshots compete for cores.
    # Whether 1 or 50 snapshots remain, all cores stay busy.
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as shared_pool:
        for family, vmem_path, snap_dir in work:
            print(f"\n[+] Processing: {vmem_path}")
            print(f"    Family: {family}  |  Output: {snap_dir}")
            try:
                run_analysis(family, vmem_path, snap_dir, executor=shared_pool)
                done += 1
            except Exception as e:
                print(f"[!] Failed: {e}")
                failed += 1
            print(f"    [{done + failed}/{len(work)}] completed")

    print(f"\n[✓] Batch complete — done: {done}, skipped: {skipped}, failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Volatility 3 ransomware feature extractor")
    parser.add_argument("--family",     help="Ransomware family name (e.g. WannaCry)")
    parser.add_argument("--vmem",       help="Path to .vmem file")
    parser.add_argument("--output-dir", help="Output directory (skips auto-naming)")
    parser.add_argument("--batch-dir",  help="Scan a directory tree for .vmem files and process all")
    args = parser.parse_args()

    # ── Batch mode: scan a session/root directory for vmem files ─────────────
    if args.batch_dir:
        batch_mode(args.batch_dir)
        return

    # ── Single file mode (called from ps1) ───────────────────────────────────
    if args.family and args.vmem:
        if args.family not in FAMILY_PROCESS_NAMES:
            print(f"[!] Unknown family '{args.family}'. Add it to FAMILY_PROCESS_NAMES.")
            exit(1)
        output_dir = args.output_dir or f"vol3_output_{args.family}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_analysis(args.family, args.vmem, output_dir)
        return

    # ── Interactive mode (standalone use) ─────────────────────────────────────
    family  = prompt_family()
    version = input("Enter snapshot number or name token (e.g. 14): ").strip()
    loops_s = input("Enter number of snapshots to process (default 5): ").strip()

    try:
        start_version = int(version)
        start_loops   = int(loops_s) if loops_s else 5
    except ValueError:
        print("[-] Invalid snapshot number. Please enter an integer.")
        return

    for i in range(start_loops):
        current_version = str(start_version + i)
        run_ts          = datetime.now().strftime("%Y%m%d_%H%M%S")

        memory_image = resolve_memory_image(family, current_version)
        if not memory_image:
            print(f"[-] vmem not found for snapshot {current_version} — skipping.")
            continue

        output_dir = args.output_dir or f"vol3_output_{family}_V{current_version}_{run_ts}"
        run_analysis(family, memory_image, output_dir)


if __name__ == "__main__":
    main()
