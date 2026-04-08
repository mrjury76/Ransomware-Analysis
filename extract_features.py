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

#cmdline indicators
NCODED_PAT = re.compile(r"-enc|encodedcommand|base64", re.I)
SUS_ARG_PAT = re.compile(r"-enc|-nop|base64|bypass|hidden|downloadstring", re.I)
SCRIPT_EXEC_PAT = re.compile(r"\.ps1|\.vbs|\.js|\.bat|\.cmd", re.I)
UNUSUAL_DIR_PAT = re.compile(r"appdata|temp|users\\public|programdata", re.I)
SCRIPT_TOOL_PAT = re.compile(r"powershell|cmd\.exe|wscript|cscript|mshta|python", re.I)

# Non-system DLL path prefixes (flags DLLs loaded from unusual locations)
SYSTEM_DLL_PATHS = {"\\windows\\system32", "\\windows\\syswow64",
                    "\\windows\\winsxs"}

CRYPTO_LIBS = {"bcrypt.dll", "crypt32.dll", "ncrypt.dll", "advapi32.dll"}
SUSPICIOUS_PORTS = {4444, 1337, 8080, 9001}

# Known ransomware encrypted file extensions
ENCRYPTED_EXTENSIONS = {".wncry", ".cerber", ".cerber2", ".cerber3",
                        ".jigsaw", ".fun", ".btc", ".encrypted",
                        ".locked", ".petya", ".petrwrap",
                        ".dharma", ".wallet", ".arena", ".adobe",
                        ".java", ".id", ".email", ".zzzzz", ".2023",
                        ".9aee"}


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

def safe_float(val, default=0.0):
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default

def nonempty_text(val):
    txt = str(val).strip()
    return txt if txt not in {"", "N/A", "0", "None", "nan"} else ""

def is_true(val):
    return str(val).strip().lower() in {"true", "1", "yes"}

def is_false(val):
    return str(val).strip().lower() in {"false", "0", "no"}

def avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def max_or_zero(values):
    return max(values) if values else 0
    
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
    child_counts = defaultdict(int)
    runtimes = []
    parent_map = {}
    
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        ppid = safe_int(r.get("PPID", 0))
        if pid:
            pids.add(pid)
            parent_map[pid] = ppid
        if ppid:
            child_counts[ppid] += 1
            
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

        create_raw = nonempty_text(r.get("CreateTime", ""))
        exit_raw = nonempty_text(r.get("ExitTime", ""))
        if exit_raw:
            exited_count += 1
        if create_raw and exit_raw:
            try:
                from datetime import datetime
                create_dt = datetime.fromisoformat(create_raw.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(exit_raw.replace("Z", "+00:00"))
                runtime = (exit_dt - create_dt).total_seconds()
                if runtime >= 0:
                    runtimes.append(runtime)
            except Exception:
                pass

    child_values = list(child_counts.value())
    hidden_parent_count = sum(1 for ppid in child_counts if ppid and ppid not in pids)

    return {
        "pslist_count":        len(pids),
        "pslist_unique_names": len(names),
        "pslist_avg_threads":  round(sum(threads) / len(threads), 2) if threads else 0,
        "pslist_avg_handles":  round(sum(handles) / len(handles), 2) if handles else 0,
        "pslist_wow64_count":  wow64_count,
        "pslist_exited_count": exited_count,
        "_pslist_pids":        pids,   # internal, stripped before output

        "pslist_avg_runtimes": avg(runtimes),
        "pslist_max_runtimes": round(max_or_zero(runtimes), 2),
        "pslist_avg_children": avg(child_values),
        "pslist_max_children": max_or_zero(child_values),
        "pslist_parent_missing_count": hidden_parent_count,
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

    cmd_lengths = []
    sus_args_count = 0
    script_exec_count = 0
    unusual_dir_count = 0
    script_tool_count = 0
    encoded_count = 0
    
    for r in rows:
        args = r.get("Args", "") or ""
        args_norm = args.strip()
        args_lower = args_norm.lower()

        if args_norm and args_norm.lower() not in {"n/a", "required memory at", ""}:
            has_args += 1
            cmd_lengths.append(len(args_norm))

            if any(tok in args_lower for tok in SUSPICIOUS_ARGS):
                suspicious += 1
            if SUS_ARG_PAT.search(args_norm):
                sus_args_count += 1
            if SCRIPT_EXEC_PAT.search(args_norm):
                script_exec_count += 1
            if UNUSUAL_DIR_PAT.search(args_norm):
                unusual_dir_count += 1
            if SCRIPT_TOOL_PAT.search(args_norm):
                script_tool_count += 1
            if ENCODED_PAT.search(args_norm):
                encoded_count += 1
    return {
        "cmdline_count":            total,
        "cmdline_with_args":        has_args,
        "cmdline_suspicious_count": suspicious,
        "cmdline_avg_length": avg(cmd_lengths),
        "cmdline_max_length": max_or_zero(cmd_lengths),
        "cmdline_sus_args_count": sus_args_count,
        "cmdline_script_exec_count": script_exec_count,
        "cmdline_unusual_dir_count": unusual_dir_count,
        "cmdline_script_tool_count": script_tool_count,
        "cmdline_encoded_count": encoded_count,
    }


def feat_dlllist(rows):
    if not rows:
        return {}
    pids = defaultdict(set)
    dll_names = set()
    non_system = 0

    load_totals_by_pid = defaultdict(float)
    load_max_by_pid = defaultdict(float)
    unusual_path_pids = set()
    unusual_path_hits = 0
    crypto_pids = set()
    crypto_hits = 0
    
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        name = r.get("Name", "").strip().lower()
        path = r.get("Path", "").strip().lower()
        load_count = safe_float(r.get("LoadCount", 0), 0.0)

        if name:
            dll_names.add(name)
        if pid:
            pids[pid].add(name)
        if path and not any(path.startswith(s) for s in SYSTEM_DLL_PATHS):
            non_system += 1
            
        if pid: #load behaviour 
            load_totals_by_pid[pid] += load_count
            if load_count > load_max_by_pid[pid]:
                load_max_by_pid[pid] = load_count

            if path and UNUSUAL_DIR_PAT.search(path):
                unusual_path_pids.add(pid)
                unusual_path_hits += 1

            if name in CRYPTO_LIBS:
                crypto_pids.add(pid)
                crypto_hits += 1

    counts = [len(v) for v in pids.values()]
    total_load_values = list(load_totals_by_pid.values())
    max_load_values = list(load_max_by_pid.values())
    
    return {
        "dlllist_total":           sum(counts),
        "dlllist_unique_dlls":     len(dll_names),
        "dlllist_avg_per_process": avg(counts),
        "dlllist_non_system":      non_system,

        "dlllist_avg_load_total_per_process": avg(total_load_values),
        "dlllist_max_load_total_per_process": round(max_or_zero(total_load_values), 2),
        "dlllist_avg_max_loadcount_per_process": avg(max_load_values),
        "dlllist_unusual_path_pid_count": len(unusual_path_pids),
        "dlllist_unusual_path_hit_count": unusual_path_hits,
        "dlllist_crypto_pid_count": len(crypto_pids),
        "dlllist_crypto_hit_count": crypto_hits,
    }

def feat_ldrmodules(rows):
    if not rows:
        return {}
    not_in_load = not_in_init = not_in_mem = hidden = 0
    pids = defaultdict(int)
    suspicious_path_pids = set()
    suspicious_path_hits = 0

    for r in rows:
        pid = safe_int(r.get("PID", r.get("Pid", 0)))
        in_load = is_true(r.get("InLoad", "True"))
        in_init = is_true(r.get("InInit", "True"))
        in_mem  = is_true(r.get("InMem",  "True"))
        path    = r.get("MappedPath", "").strip()

        if pid:
            pids[pid] += 1
        if not in_load:
            not_in_load += 1
        if not in_init:
            not_in_init += 1
        if not in_mem:
            not_in_mem += 1
        if (not in_load or not in_init or not in_mem) and not path:
            hidden += 1
        if path and UNUSUAL_DIR_PAT.search(path.lower()):
            suspicious_path_pids.add(pid)
            suspicious_path_hits += 1

    module_counts = list(pids.values())
    return {
        "ldrmodules_total":        len(rows),
        "ldrmodules_not_in_load":  not_in_load,
        "ldrmodules_not_in_init":  not_in_init,
        "ldrmodules_not_in_mem":   not_in_mem,
        "ldrmodules_hidden_count": hidden,
        "ldrmodules_avg_per_process": avg(module_counts),
        "ldrmodules_max_per_process": max_or_zero(module_counts),
        "ldrmodules_suspicious_path_pid_count": len({pid for pid in suspicious_path_pids if pid}),
        "ldrmodules_suspicious_path_hit_count": suspicious_path_hits,
    }


def feat_vadinfo(rows):
    if not rows:
        return {}
    total = len(rows)
    exec_count = private_exec = 0
    pids = defaultdict(int)
    pid_sizes = defaultdict(float)
    pid_max_region = defaultdict(float)
    pid_rwx = defaultdict(int)
    pid_private = defaultdict(int)
    
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        prot = (r.get("Protection", "") or "").strip()
        private = is_true(r.get("PrivateMemory", "")) or "private" in (r.get("Tag", "") or "").lower()
        size = safe_float(r.get("Size", 0), 0.0)

        if prot in EXEC_PROTECTIONS:
            exec_count += 1
            if private:
                private_exec += 1
        if pid:
            pids[pid] += 1
            pid_sizes[pid] += size
            if size > pid_max_region[pid]:
                pid_max_region[pid] = size
            if "RWX" in prot.upper() or "EXECUTE_READWRITE" in prot.upper():
                pid_rwx[pid] += 1
            if private:
                pid_private[pid] += 1

    vads_per_pid = list(pids.values())
    total_sizes = list(pid_sizes.values())
    max_region_sizes = list(pid_max_region.values())
    private_counts = list(pid_private.values())
    rwx_counts = list(pid_rwx.values())
    
    return {
        "vad_total":           total,
        "vad_exec_count":      exec_count,
        "vad_private_exec":    private_exec,
        "vad_pid_count": len(pids),
        "vad_avg_regions_per_process": avg(vads_per_pid),
        "vad_max_regions_per_process": max_or_zero(vads_per_pid),
        "vad_avg_total_mem_per_process": round(avg(total_sizes), 2) if total_sizes else 0,
        "vad_max_total_mem_per_process": round(max_or_zero(total_sizes), 2),
        "vad_avg_max_region_size_per_process": round(avg(max_region_sizes), 2) if max_region_sizes else 0,
        "vad_private_pid_count": sum(1 for v in private_counts if v > 0),
        "vad_private_region_count": sum(private_counts),
        "vad_rwx_pid_count": sum(1 for v in rwx_counts if v > 0),
        "vad_rwx_region_count": sum(rwx_counts),
    }


def feat_malfind(rows):
    if not rows:
        return {}
    total = len(rows)
    exe_regions = private_count = 0
    pids = defaultdict(int)
    rwx_pids = set()
    rwx_regions = 0
    
    for r in rows:
        pid = safe_int(r.get("PID", 0))
        prot = (r.get("Protection", "") or "").strip()
        private = is_true(r.get("PrivateMemory", ""))
        hexdump = (r.get("Hexdump", "") or "").strip()
        
        # MZ header in dump = likely injected PE
        if "4d 5a" in hexdump.lower() or prot == "PAGE_EXECUTE_READWRITE":
            exe_regions += 1
        if private:
            private_count += 1
        if pid:
            pids[pid] += 1
        if "RWX" in prot.upper() or "EXECUTE_READWRITE" in prot.upper():
            rwx_regions += 1
            if pid:
                rwx_pids.add(pid)

    regions_per_pid = list(pids.values())
    
    return {
        "malfind_count":       total,
        "malfind_exe_regions": exe_regions,
        "malfind_private":     private_count,
        "malfind_pid_count": len(pids),
        "malfind_avg_regions_per_process": avg(regions_per_pid),
        "malfind_max_regions_per_process": max_or_zero(regions_per_pid),
        "malfind_rwx_pid_count": len(rwx_pids),
        "malfind_rwx_region_count": rwx_regions,
        "malfind_injected_pid_count": len(pids),  # any malfind hit -> suspicious/injected
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
    pids = defaultdict(int)
    outbound_pids = set()
    outbound_count = 0
    suspicious_port_pids = set()
    suspicious_port_hits = 0
    tcp_pids = set()
    udp_pids = set()
    established_pids = set()
    listening_pids = set()

    remote_ips_by_pid = defaultdict(set)

    for r in rows:
        pid = safe_int(r.get("PID", 0))
        state = r.get("State", "").strip().upper()
        remote = r.get("ForeignAddr", "").strip()
        proto = r.get("Proto", "").strip().upper()
        foreign_port = safe_int(r.get("ForeignPort", 0), 0)
        local_port = safe_int(r.get("LocalPort", 0), 0)

        if "ESTABLISHED" in state:
            established += 1
            if pid:
                established_pids.add(pid)
        if "LISTEN" in state:
            listening += 1
            if pid:
                listening_pids.add(pid)
        if remote and remote not in {"", "0.0.0.0", "*", "N/A"}:
            remote_ips.add(remote)
            if pid:
                remote_ips_by_pid[pid].add(remote)
        if pid:
            pids[pid] += 1

        outbound = remote not in {"", "0.0.0.0", "*", "N/A"}
        if outbound:
            outbound_count += 1
            if pid:
                outbound_pids.add(pid)

        if foreign_port in SUSPICIOUS_PORTS or local_port in SUSPICIOUS_PORTS:
            suspicious_port_hits += 1
            if pid:
                suspicious_port_pids.add(pid)

        if "TCP" in proto and pid:
            tcp_pids.add(pid)
        if "UDP" in proto and pid:
            udp_pids.add(pid)

    conns_per_pid = list(pids.values())
    unique_remote_ips_per_pid = [len(v) for v in remote_ips_by_pid.values()]

    return {
        "netstat_total":       len(rows),
        "netstat_established": established,
        "netstat_listening":   listening,
        "netstat_unique_ips":  len(remote_ips),
        "netstat_pid_count": len(pids),
        "netstat_avg_connections_per_process": avg(conns_per_pid),
        "netstat_max_connections_per_process": max_or_zero(conns_per_pid),
        "netstat_avg_unique_ips_per_process": avg(unique_remote_ips_per_pid),
        "netstat_outbound_count": outbound_count,
        "netstat_outbound_pid_count": len(outbound_pids),
        "netstat_suspicious_port_hit_count": suspicious_port_hits,
        "netstat_suspicious_port_pid_count": len(suspicious_port_pids),
        "netstat_tcp_pid_count": len(tcp_pids),
        "netstat_udp_pid_count": len(udp_pids),
        "netstat_established_pid_count": len(established_pids),
        "netstat_listening_pid_count": len(listening_pids),
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

    # ── Ratio features (normalize across system load) ──────────────────────
    vad_total = features.get("vad_total", 0)
    if vad_total > 0:
        features["vad_exec_ratio"]     = round(features.get("vad_exec_count", 0) / vad_total, 4)
        features["vad_priv_exec_ratio"] = round(features.get("vad_private_exec", 0) / vad_total, 4)
        features["vad_rwx_ratio"] = round(features.get("vad_rwx_region_count", 0) / vad_total, 4)
        features["vad_private_ratio"] = round(features.get("vad_private_region_count", 0) / vad_total, 4)
    else:
        features["vad_exec_ratio"]     = 0
        features["vad_priv_exec_ratio"] = 0
        features["vad_rwx_ratio"] = 0
        features["vad_private_ratio"] = 0

    handle_total = features.get("handle_total", 0)
    if handle_total > 0:
        features["handle_file_ratio"]     = round(features.get("handle_file_count", 0) / handle_total, 4)
        features["handle_registry_ratio"] = round(features.get("handle_registry_count", 0) / handle_total, 4)
        features["handle_mutex_ratio"]    = round(features.get("handle_mutex_count", 0) / handle_total, 4)
    else:
        features["handle_file_ratio"]     = 0
        features["handle_registry_ratio"] = 0
        features["handle_mutex_ratio"]    = 0

    ldr_total = features.get("ldrmodules_total", 0)
    if ldr_total > 0:
        features["ldrmodules_hidden_ratio"] = round(features.get("ldrmodules_hidden_count", 0) / ldr_total, 4)
        features["ldrmodules_suspicious_path_ratio"] = round(features.get("ldrmodules_suspicious_path_hit_count", 0) / ldr_total, 4)
    else:
        features["ldrmodules_hidden_ratio"] = 0
        features["ldrmodules_suspicious_path_ratio"] = 0
        
    filescan_total = features.get("filescan_total", 0)
    if filescan_total > 0:
        features["filescan_encrypted_ratio"] = round(features.get("filescan_encrypted", 0) / filescan_total, 4)
    else:
        features["filescan_encrypted_ratio"] = 0

    netstat_total = features.get("netstat_total", 0)
    if netstat_total > 0:
        features["netstat_established_ratio"] = round(features.get("netstat_established", 0) / netstat_total, 4)
        features["netstat_outbound_ratio"] = round(features.get("netstat_outbound_count", 0) / netstat_total, 4)
        features["netstat_suspicious_port_ratio"] = round(features.get("netstat_suspicious_port_hit_count", 0) / netstat_total, 4)
    else:
        features["netstat_established_ratio"] = 0
        features["netstat_outbound_ratio"] = 0
        features["netstat_suspicious_port_ratio"] = 0

    cmdline_total = features.get("cmdline_count", 0)
    if cmdline_total > 0:
        features["cmdline_encoded_ratio"] = round(features.get("cmdline_encoded_count", 0) / cmdline_total, 4)
        features["cmdline_script_exec_ratio"] = round(features.get("cmdline_script_exec_count", 0) / cmdline_total, 4)
        features["cmdline_unusual_dir_ratio"] = round(features.get("cmdline_unusual_dir_count", 0) / cmdline_total, 4)
    else:
        features["cmdline_encoded_ratio"] = 0
        features["cmdline_script_exec_ratio"] = 0
        features["cmdline_unusual_dir_ratio"] = 0

    dll_total = features.get("dlllist_total", 0)
    if dll_total > 0:
        features["dlllist_unusual_path_ratio"] = round(features.get("dlllist_unusual_path_hit_count", 0) / dll_total, 4)
        features["dlllist_crypto_ratio"] = round(features.get("dlllist_crypto_hit_count", 0) / dll_total, 4)
    else:
        features["dlllist_unusual_path_ratio"] = 0
        features["dlllist_crypto_ratio"] = 0

    malfind_total = features.get("malfind_count", 0)
    if malfind_total > 0:
        features["malfind_rwx_ratio"] = round(features.get("malfind_rwx_region_count", 0) / malfind_total, 4)
    else:
        features["malfind_rwx_ratio"] = 0
        
    # ── Behavior-based stage label ──────────────────────────────────────────
    # Assigns stage from observable indicators rather than time elapsed.
    # Used as an alternative to stage_hint (time-based).
    encrypted     = features.get("filescan_encrypted", 0)
    malfind_exe   = features.get("malfind_exe_regions", 0)
    malfind_count = features.get("malfind_count", 0)

    if encrypted == 0 and malfind_exe == 0 and malfind_count == 0:
        behavior_stage = 0   # baseline — no malware indicators
    elif encrypted == 0:
        behavior_stage = 1   # executing — injection/malfind but no encryption yet
    elif encrypted < 100:
        behavior_stage = 2   # encrypting — files being encrypted
    else:
        behavior_stage = 3   # post-encryption — heavy encryption observed

    row = {
        "family":           meta.get("family", ""),
        "stage_hint":       meta.get("stage_hint", ""),
        "behavior_stage":   behavior_stage,
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
    meta_cols = ["family", "stage_hint", "behavior_stage", "actual_offset_s",
                 "target_offset_s", "rep", "run", "snap_name", "snap_dir"]
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
