"""
extract_features.py
-------------------
Walks a VMSnapshots directory tree, reads per-plugin CSVs from each snapshot
folder, and outputs a single feature-matrix CSV ready for ML training.

Usage:
    python3 extract_features.py --scan-dir /mnt/d/Patrick/VMSnapshots
    python3 extract_features.py --scan-dir /mnt/d/Patrick/VMSnapshots/WannaCry_20260326_140000
    python3 extract_features.py --scan-dir /mnt/d/Patrick/VMSnapshots --out features.csv
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

# Executable VAD protection strings
EXEC_PROTECTIONS = {"PAGE_EXECUTE", "PAGE_EXECUTE_READ",
                    "PAGE_EXECUTE_READWRITE", "PAGE_EXECUTE_WRITECOPY"}

# Suspicious cmdline tokens
SUSPICIOUS_ARGS = {"/c", "/q", "cmd", "powershell", "wscript", "cscript",
                   "regsvr32", "rundll32", "mshta", "certutil", "bitsadmin"}

# Non-system DLL path prefixes (flags DLLs loaded from unusual locations)
SYSTEM_DLL_PATHS = {"\\windows\\system32", "\\windows\\syswow64",
                    "\\windows\\winsxs"}

# Known ransomware encrypted file extensions
ENCRYPTED_EXTENSIONS = {".wncry", ".cerber", ".cerber2", ".cerber3",
                        ".jigsaw", ".fun", ".btc", ".encrypted",
                        ".locked", ".petya", ".petrwrap"}


def read_csv(path):
    """Return list of dicts from a CSV, or empty list if missing/broken."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def safe_int(val, default=0):
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def is_true(val):
    return str(val).strip().lower() in {"true", "1", "yes"}


def is_false(val):
    return str(val).strip().lower() in {"false", "0", "no"}


# =============================================================================
# Per-plugin feature extractors
# =============================================================================

def feat_pslist(rows):
    if not rows:
        return {}
    pids = set()
    names = set()
    threads, handles = [], []
    wow64_count = exited_count = 0
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        if pid:
            pids.add(pid)
        name = r.get("ImageFileName", "").strip()
        if name:
            names.add(name.lower())
        t = safe_int(r.get("Threads", 0))
        if t:
            threads.append(t)
        h = safe_int(r.get("Handles", 0))
        if h:
            handles.append(h)
        if is_true(r.get("Wow64", "")):
            wow64_count += 1
        if r.get("ExitTime", "").strip() not in {"", "N/A", "0"}:
            exited_count += 1
    return {
        "pslist_count":        len(pids),
        "pslist_unique_names": len(names),
        "pslist_avg_threads":  round(sum(threads) / len(threads), 2) if threads else 0,
        "pslist_avg_handles":  round(sum(handles) / len(handles), 2) if handles else 0,
        "pslist_wow64_count":  wow64_count,
        "pslist_exited_count": exited_count,
        "_pslist_pids":        pids,   # internal, stripped before output
    }


def feat_psscan(rows, pslist_pids):
    if not rows:
        return {"psscan_count": 0, "hidden_process_count": 0}
    pids = set()
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        if pid:
            pids.add(pid)
    hidden = len(pids - pslist_pids)
    return {
        "psscan_count":        len(pids),
        "hidden_process_count": hidden,
    }


def feat_cmdline(rows):
    if not rows:
        return {}
    total = len(rows)
    has_args = suspicious = 0
    for r in rows:
        args = r.get("Args", "").strip().lower()
        if args and args not in {"n/a", "required memory at", ""}:
            has_args += 1
            if any(tok in args for tok in SUSPICIOUS_ARGS):
                suspicious += 1
    return {
        "cmdline_count":            total,
        "cmdline_with_args":        has_args,
        "cmdline_suspicious_count": suspicious,
    }


def feat_dlllist(rows):
    if not rows:
        return {}
    pids = defaultdict(set)
    dll_names = set()
    non_system = 0
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        name = r.get("Name", "").strip().lower()
        path = r.get("Path", "").strip().lower()
        if name:
            dll_names.add(name)
        if pid:
            pids[pid].add(name)
        if path and not any(path.startswith(s) for s in SYSTEM_DLL_PATHS):
            non_system += 1
    counts = [len(v) for v in pids.values()]
    return {
        "dlllist_total":           sum(counts),
        "dlllist_unique_dlls":     len(dll_names),
        "dlllist_avg_per_process": round(sum(counts) / len(counts), 2) if counts else 0,
        "dlllist_non_system":      non_system,
    }


def feat_ldrmodules(rows):
    if not rows:
        return {}
    not_in_load = not_in_init = not_in_mem = hidden = 0
    for r in rows:
        in_load = is_true(r.get("InLoad", "True"))
        in_init = is_true(r.get("InInit", "True"))
        in_mem  = is_true(r.get("InMem",  "True"))
        path    = r.get("MappedPath", "").strip()
        if not in_load:
            not_in_load += 1
        if not in_init:
            not_in_init += 1
        if not in_mem:
            not_in_mem += 1
        if (not in_load or not in_init or not in_mem) and not path:
            hidden += 1
    return {
        "ldrmodules_total":        len(rows),
        "ldrmodules_not_in_load":  not_in_load,
        "ldrmodules_not_in_init":  not_in_init,
        "ldrmodules_not_in_mem":   not_in_mem,
        "ldrmodules_hidden_count": hidden,
    }


def feat_vadinfo(rows):
    if not rows:
        return {}
    total = len(rows)
    exec_count = private_exec = 0
    for r in rows:
        prot    = r.get("Protection", "").strip()
        private = is_true(r.get("PrivateMemory", ""))
        if prot in EXEC_PROTECTIONS:
            exec_count += 1
            if private:
                private_exec += 1
    return {
        "vad_total":           total,
        "vad_exec_count":      exec_count,
        "vad_private_exec":    private_exec,
    }


def feat_malfind(rows):
    if not rows:
        return {}
    total = len(rows)
    exe_regions = private_count = 0
    for r in rows:
        prot    = r.get("Protection", "").strip()
        private = is_true(r.get("PrivateMemory", ""))
        hexdump = r.get("Hexdump", "").strip()
        # MZ header in dump = likely injected PE
        if "4d 5a" in hexdump.lower() or prot == "PAGE_EXECUTE_READWRITE":
            exe_regions += 1
        if private:
            private_count += 1
    return {
        "malfind_count":       total,
        "malfind_exe_regions": exe_regions,
        "malfind_private":     private_count,
    }


def feat_handles(rows):
    if not rows:
        return {}
    type_counts = defaultdict(int)
    encrypted_handles = 0
    for r in rows:
        htype = r.get("Type", "").strip()
        name  = r.get("Name", "").strip().lower()
        type_counts[htype] += 1
        ext = os.path.splitext(name)[1]
        if ext in ENCRYPTED_EXTENSIONS:
            encrypted_handles += 1
    return {
        "handle_total":           len(rows),
        "handle_file_count":      type_counts.get("File", 0),
        "handle_registry_count":  type_counts.get("Key", 0),
        "handle_mutex_count":     type_counts.get("Mutant", 0),
        "handle_process_count":   type_counts.get("Process", 0),
        "handle_thread_count":    type_counts.get("Thread", 0),
        "handle_encrypted_files": encrypted_handles,
    }


def feat_filescan(rows):
    if not rows:
        return {}
    total = len(rows)
    encrypted = 0
    for r in rows:
        name = r.get("Name", "").strip().lower()
        ext  = os.path.splitext(name)[1]
        if ext in ENCRYPTED_EXTENSIONS:
            encrypted += 1
    return {
        "filescan_total":     total,
        "filescan_encrypted": encrypted,
    }


def feat_svcscan(rows):
    if not rows:
        return {}
    running = stopped = 0
    for r in rows:
        state = r.get("State", "").strip().upper()
        if "RUNNING" in state:
            running += 1
        elif "STOPPED" in state:
            stopped += 1
    return {
        "svcscan_total":   len(rows),
        "svcscan_running": running,
        "svcscan_stopped": stopped,
    }


def feat_privileges(rows):
    if not rows:
        return {}
    sedebug = enabled = 0
    for r in rows:
        attrs = r.get("Attributes", "").strip().lower()
        name  = r.get("Privileges", r.get("Name", "")).strip().lower()
        if "present" in attrs or "enabled" in attrs:
            enabled += 1
            if "debug" in name:
                sedebug += 1
    return {
        "priv_total_enabled": enabled,
        "priv_sedebug_count": sedebug,
    }


def feat_netstat(rows):
    if not rows:
        return {}
    established = listening = 0
    remote_ips  = set()
    for r in rows:
        state  = r.get("State", "").strip().upper()
        remote = r.get("ForeignAddr", "").strip()
        if "ESTABLISHED" in state:
            established += 1
        if "LISTEN" in state:
            listening += 1
        if remote and remote not in {"", "0.0.0.0", "*", "N/A"}:
            remote_ips.add(remote)
    return {
        "netstat_total":       len(rows),
        "netstat_established": established,
        "netstat_listening":   listening,
        "netstat_unique_ips":  len(remote_ips),
    }


# =============================================================================
# Snapshot processing
# =============================================================================

PLUGIN_FILES = {
    "windows.pslist":     "windows.pslist.csv",
    "windows.psscan":     "windows.psscan.csv",
    "windows.cmdline":    "windows.cmdline.csv",
    "windows.dlllist":    "windows.dlllist.csv",
    "windows.ldrmodules": "windows.ldrmodules.csv",
    "windows.vadinfo":    "windows.vadinfo.csv",
    "windows.malfind":    "windows.malfind.csv",
    "windows.handles":    "windows.handles.csv",
    "windows.filescan":   "windows.filescan.csv",
    "windows.svcscan":    "windows.svcscan.csv",
    "windows.privileges": "windows.privileges.csv",
    "windows.netstat":    "windows.netstat.csv",
}


def process_snapshot(snap_dir):
    """Extract feature row from a single snapshot directory."""
    meta_path = os.path.join(snap_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Read all plugin CSVs
    plugin_rows = {}
    for plugin, fname in PLUGIN_FILES.items():
        plugin_rows[plugin] = read_csv(os.path.join(snap_dir, fname))

    # Extract features
    pslist_feats = feat_pslist(plugin_rows["windows.pslist"])
    pslist_pids  = pslist_feats.pop("_pslist_pids", set())

    features = {}
    features.update(pslist_feats)
    features.update(feat_psscan(plugin_rows["windows.psscan"], pslist_pids))
    features.update(feat_cmdline(plugin_rows["windows.cmdline"]))
    features.update(feat_dlllist(plugin_rows["windows.dlllist"]))
    features.update(feat_ldrmodules(plugin_rows["windows.ldrmodules"]))
    features.update(feat_vadinfo(plugin_rows["windows.vadinfo"]))
    features.update(feat_malfind(plugin_rows["windows.malfind"]))
    features.update(feat_handles(plugin_rows["windows.handles"]))
    features.update(feat_filescan(plugin_rows["windows.filescan"]))
    features.update(feat_svcscan(plugin_rows["windows.svcscan"]))
    features.update(feat_privileges(plugin_rows["windows.privileges"]))
    features.update(feat_netstat(plugin_rows["windows.netstat"]))

    row = {
        "family":           meta.get("family", ""),
        "stage_hint":       meta.get("stage_hint", ""),
        "actual_offset_s":  meta.get("actual_offset_s", ""),
        "target_offset_s":  meta.get("target_offset_s", ""),
        "rep":              meta.get("rep", ""),
        "run":              meta.get("run", ""),
        "snap_name":        meta.get("snap_name", ""),
        "snap_dir":         snap_dir,
    }
    row.update(features)
    return row


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract ML features from Volatility snapshot CSVs")
    parser.add_argument("--scan-dir", required=True,
                        help="Root directory to scan for snapshot folders (containing meta.json)")
    parser.add_argument("--out", default="features.csv",
                        help="Output CSV path (default: features.csv)")
    args = parser.parse_args()

    # Find all snapshot directories (those with meta.json)
    snap_dirs = []
    for root, _, files in os.walk(args.scan_dir):
        if "meta.json" in files:
            snap_dirs.append(root)

    if not snap_dirs:
        print(f"[-] No snapshot directories (meta.json) found under {args.scan_dir}")
        return

    print(f"[+] Found {len(snap_dirs)} snapshot(s) to process")

    rows = []
    for i, snap_dir in enumerate(sorted(snap_dirs), 1):
        print(f"  [{i}/{len(snap_dirs)}] {snap_dir}")
        row = process_snapshot(snap_dir)
        if row:
            rows.append(row)

    if not rows:
        print("[-] No feature rows extracted.")
        return

    # Build consistent field order: metadata first, then features
    meta_cols = ["family", "stage_hint", "actual_offset_s", "target_offset_s",
                 "rep", "run", "snap_name", "snap_dir"]
    feat_cols = [k for k in rows[0] if k not in meta_cols]
    fieldnames = meta_cols + feat_cols

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✓] Features saved to: {args.out}")
    print(f"[✓] Rows: {len(rows)}  |  Feature columns: {len(feat_cols)}")


if __name__ == "__main__":
    main()
