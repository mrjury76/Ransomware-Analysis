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
ENCODED_PAT = re.compile(r"-enc|encodedcommand|base64", re.I)
SUS_ARG_PAT = re.compile(r"-enc|-nop|base64|bypass|hidden|downloadstring", re.I)
SCRIPT_EXEC_PAT = re.compile(r"\.ps1|\.vbs|\.js|\.bat|\.cmd", re.I)
UNUSUAL_DIR_PAT = re.compile(r"appdata|temp|users\\public|programdata", re.I)
SCRIPT_TOOL_PAT = re.compile(r"powershell|cmd\.exe|wscript|cscript|mshta|python", re.I)

# Ransomware-specific cmdline tokens (strong stage indicators per PERD/RENTAKA)
RANSOM_CMDLINE_TOKENS = {"vssadmin", "delete", "shadows", "bcdedit",
                         "wbadmin", "recoveryenabled", "bootstatuspolicy",
                         "wmic", "shadowcopy"}

# Non-system DLL path prefixes (flags DLLs loaded from unusual locations)
SYSTEM_DLL_PATHS = {"\\windows\\system32", "\\windows\\syswow64",
                    "\\windows\\winsxs"}

CRYPTO_LIBS = {"bcrypt.dll", "crypt32.dll", "ncrypt.dll", "advapi32.dll"}
SUSPICIOUS_PORTS = {4444, 1337, 8080, 9001}

# Crypto-related DLLs (loading these indicates cryptographic setup)
CRYPTO_DLLS = {"advapi32.dll", "crypt32.dll", "bcrypt.dll", "ncrypt.dll",
               "rsaenh.dll", "cryptsp.dll", "cryptbase.dll"}

# Known ransomware encrypted file extensions
ENCRYPTED_EXTENSIONS = {".wncry", ".cerber", ".cerber2", ".cerber3",
                        ".jigsaw", ".fun", ".btc", ".encrypted",
                        ".locked", ".petya", ".petrwrap",
                        ".dharma", ".wallet", ".arena", ".adobe",
                        ".java", ".id", ".email", ".zzzzz", ".2023",
                        ".9aee"}

# Substrings in filenames that indicate encryption activity (not just extensions)
ENCRYPTED_SUBSTRINGS = {"wncry", ".cerber", ".dharma", ".2023", ".9aee",
                        ".encrypted", ".locked"}

# Ransom note filenames (family-agnostic indicators of encryption completion)
RANSOM_NOTE_NAMES = {"@please_read_me@", "readme_for_decrypt", "_readme_",
                     "help_decrypt", "how_to_decrypt", "restore_files",
                     "decrypt_instruction", "your_files_are_encrypted",
                     "recover_your_files"}

# Process names that indicate ransomware pre-encryption behavior
RANSOM_PROCESS_NAMES = {"vssadmin.exe", "wbadmin.exe", "bcdedit.exe",
                        "wmic.exe"}


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

    child_values = list(child_counts.values())
    hidden_parent_count = sum(1 for ppid in child_counts if ppid and ppid not in pids)

    ransom_procs = sum(1 for n in names if n in RANSOM_PROCESS_NAMES)
    return {
        "pslist_count":        len(pids),
        "pslist_unique_names": len(names),
        "pslist_avg_threads":  round(sum(threads) / len(threads), 2) if threads else 0,
        # "pslist_avg_handles": always 0 — Handles col missing from pslist CSV
        "pslist_wow64_count":  wow64_count,
        "pslist_exited_count": exited_count,
        "pslist_ransom_procs": ransom_procs,
        "_pslist_pids":        pids,   # internal, stripped before output

        # "pslist_avg_runtimes": always 0 — CreateTime/ExitTime not in pslist CSV
        # "pslist_max_runtimes": always 0 — same reason
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
    unusual_dir_count = 0
    script_tool_count = 0

    # Patterns below never fire on current data — kept for future reference:
    # SUS_ARG_PAT, ENCODED_PAT, RANSOM_CMDLINE_TOKENS, SCRIPT_EXEC_PAT

    for r in rows:
        args = r.get("Args", "") or ""
        args_norm = args.strip()
        args_lower = args_norm.lower()

        if args_norm and args_norm.lower() not in {"n/a", "required memory at", ""}:
            has_args += 1
            cmd_lengths.append(len(args_norm))

            if any(tok in args_lower for tok in SUSPICIOUS_ARGS):
                suspicious += 1
            if UNUSUAL_DIR_PAT.search(args_norm):
                unusual_dir_count += 1
            if SCRIPT_TOOL_PAT.search(args_norm):
                script_tool_count += 1
    return {
        "cmdline_count":            total,
        "cmdline_with_args":        has_args,
        "cmdline_suspicious_count": suspicious,
        "cmdline_avg_length":       avg(cmd_lengths),
        "cmdline_max_length":       max_or_zero(cmd_lengths),
        "cmdline_unusual_dir_count": unusual_dir_count,
        "cmdline_script_tool_count": script_tool_count,
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

    crypto_loaded = sum(1 for d in dll_names if d in CRYPTO_DLLS)
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
        "dlllist_crypto_dlls":     crypto_loaded,
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
        "ldrmodules_hidden_count": hidden,               # always 0
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
        "vad_avg_total_mem_per_process": round(avg(total_sizes), 2) if total_sizes else 0,    # always 0 -- Size col missing
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
    mz_regions = 0        # injected PE header (MZ) in hexdump
    rwx_regions = 0       # PAGE_EXECUTE_READWRITE (writable shellcode staging)
    private_exec = 0      # private + any executable prot (process hollowing pattern)
    private_count = 0
    shellcode_regions = 0 # disasm starts with call/jmp = shellcode entry
    pids = defaultdict(int)
    rwx_pids = set()

    for r in rows:
        pid = safe_int(r.get("PID", 0))
        prot = (r.get("Protection", "") or "").strip().upper()
        private = is_true(r.get("PrivateMemory", ""))
        # strip spaces before searching so "4D 5A" and "4d 5a" both match
        hexdump = (r.get("Hexdump", "") or "").replace(" ", "").lower()
        disasm = (r.get("Disasm", "") or "").lower()

        if "4d5a" in hexdump:
            mz_regions += 1
        if private:
            private_count += 1
        if pid:
            pids[pid] += 1
        is_rwx = "EXECUTE_READWRITE" in prot or prot == "RWX"
        if is_rwx:
            rwx_regions += 1
            if pid:
                rwx_pids.add(pid)
        # private memory that is executable (but not necessarily writable)
        if private and "EXECUTE" in prot:
            private_exec += 1
        # disasm starting with a call or jmp strongly suggests shellcode
        first_line = disasm.split("\n")[0] if disasm else ""
        if first_line and re.search(r"\bcall\b|\bjmp\b|\bjne\b|\bjz\b", first_line):
            shellcode_regions += 1

    regions_per_pid = list(pids.values())

    return {
        "malfind_count":                    total,
        "malfind_mz_regions":               mz_regions,
        "malfind_rwx_region_count":         rwx_regions,
        "malfind_private":                  private_count,
        "malfind_private_exec":             private_exec,
        "malfind_shellcode_regions":        shellcode_regions,
        "malfind_pid_count":                len(pids),
        "malfind_avg_regions_per_process":  avg(regions_per_pid),
        "malfind_max_regions_per_process":  max_or_zero(regions_per_pid),
        "malfind_rwx_pid_count":            len(rwx_pids),
    }


def feat_handles(rows):
    if not rows:
        return {}
    type_counts = defaultdict(int)
    pids = set()
    encrypted_handles = 0
    for r in rows:
        htype = r.get("Type", "").strip()
        name  = r.get("Name", "").strip().lower()
        pid   = r.get("PID", "").strip()
        type_counts[htype] += 1
        if pid:
            pids.add(pid)
        ext = os.path.splitext(name)[1]
        if ext in ENCRYPTED_EXTENSIONS:
            encrypted_handles += 1

    total = len(rows)
    n_procs = len(pids) if pids else 1
    return {
        "handle_total":           total,
        "handle_avg_per_process": round(total / n_procs, 2),
        # existing types
        "handle_file_count":      type_counts.get("File", 0),
        "handle_registry_count":  type_counts.get("Key", 0),
        "handle_mutex_count":     type_counts.get("Mutant", 0),
        "handle_process_count":   type_counts.get("Process", 0),
        "handle_thread_count":    type_counts.get("Thread", 0),
        # new types (matching their dataset)
        "handle_port_count":      type_counts.get("ALPC Port", 0),
        "handle_event_count":     type_counts.get("Event", 0),
        "handle_section_count":   type_counts.get("Section", 0),
        "handle_semaphore_count": type_counts.get("Semaphore", 0),
        "handle_timer_count":     type_counts.get("Timer", 0) + type_counts.get("IRTimer", 0),
        "handle_desktop_count":   type_counts.get("Desktop", 0),
        "handle_directory_count": type_counts.get("Directory", 0),
        "handle_token_count":     type_counts.get("Token", 0),
        # ratios
        "handle_file_ratio":      round(type_counts.get("File", 0) / total, 4) if total else 0,
        "handle_registry_ratio":  round(type_counts.get("Key", 0)  / total, 4) if total else 0,
        "handle_mutex_ratio":     round(type_counts.get("Mutant", 0) / total, 4) if total else 0,
        "handle_encrypted_files": encrypted_handles,
    }


def feat_filescan(rows):
    if not rows:
        return {}
    total = len(rows)
    encrypted = 0
    ransom_notes = 0
    for r in rows:
        name = r.get("Name", "").strip().lower()
        # Extension-based match
        ext = os.path.splitext(name)[1]
        if ext in ENCRYPTED_EXTENSIONS:
            encrypted += 1
            continue
        # Substring-based match (catches WannaCry .WNCRYT embedded in path)
        if any(sub in name for sub in ENCRYPTED_SUBSTRINGS):
            encrypted += 1
            continue
        # Ransom note detection
        basename = name.rsplit("\\", 1)[-1] if "\\" in name else name
        if any(note in basename for note in RANSOM_NOTE_NAMES):
            ransom_notes += 1
    return {
        "filescan_total":        total,
        "filescan_encrypted":    encrypted,
        "filescan_ransom_notes": ransom_notes,
    }


SECURITY_SERVICES = {"windefend", "msmpeng", "mbamlservice", "vsserv",
                     "sophos", "mcafee", "vss", "swprv", "wbengine",
                     "sqlwriter", "sqlbrowser", "mssqlserver"}

def feat_svcscan(rows):
    if not rows:
        return {}
    # Vol3 svcscan duplicates entries across service tables -- deduplicate by
    # (name, state) so that VSS/wbengine/swprv don't inflate counts on benign machines.
    seen = set()
    running = stopped = security_stopped = 0
    for r in rows:
        state = r.get("State", "").strip().upper()
        name  = r.get("Name", r.get("ServiceName", "")).strip().lower()
        key   = (name, state)
        if key in seen:
            continue
        seen.add(key)
        if "RUNNING" in state:
            running += 1
        elif "STOPPED" in state:
            stopped += 1
            if any(svc in name for svc in SECURITY_SERVICES):
                security_stopped += 1
    return {
        "svcscan_total":            len(seen),
        "svcscan_running":          running,
        "svcscan_stopped":          stopped,
        "svcscan_security_stopped": security_stopped,
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
        # "priv_sedebug_count": sedebug,                   # always 0
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
        "netstat_suspicious_port_hit_count": suspicious_port_hits,   # always 0
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


def process_snapshot(snap_dir, use_cache=True):
    """Extract feature row from a single snapshot directory.

    Caches results to features_cache.json in the snapshot folder.
    Subsequent runs load from cache instead of re-reading plugin CSVs.
    """
    meta_path  = os.path.join(snap_dir, "meta.json")
    cache_path = os.path.join(snap_dir, "features_cache.json")

    if not os.path.exists(meta_path):
        return None

    # Return cached features if available
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

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
    # features.update(feat_netstat(plugin_rows["windows.netstat"]))

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

    # handle ratios are now computed inside feat_handles -- no recalculation needed here

    ldr_total = features.get("ldrmodules_total", 0)
    if ldr_total > 0:
        # ldrmodules_hidden_ratio: always 0 -- removed
        features["ldrmodules_suspicious_path_ratio"] = round(features.get("ldrmodules_suspicious_path_hit_count", 0) / ldr_total, 4)
    else:
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
        # netstat_suspicious_port_ratio: always 0 -- removed
    else:
        features["netstat_established_ratio"] = 0
        features["netstat_outbound_ratio"] = 0

    cmdline_total = features.get("cmdline_count", 0)
    if cmdline_total > 0:
        features["cmdline_unusual_dir_ratio"] = round(features.get("cmdline_unusual_dir_count", 0) / cmdline_total, 4)
    else:
        features["cmdline_unusual_dir_ratio"] = 0
    # cmdline_encoded_ratio, cmdline_script_exec_ratio: always 0 -- removed

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
        
    # ── Behavior-based stage label (4-class) ─────────────────────────────────
    # Grounded in ransomware lifecycle research (Kharraz 2016, Scaife 2016,
    # Sgandurra 2016, Continella 2016) and empirical analysis of universal signals.
    #
    #   0 -- Benign / Dormant
    #       No observable ransomware activity.
    #
    #   1 -- Pre-encryption active (defense evasion / recon phase)
    #       Ransomware is running and preparing -- files are still INTACT.
    #       This is the highest-value detection window.
    #       Universal signals: elevated process counts, DLL injection anomalies,
    #       suspicious handles, service manipulation.
    #
    #   2 -- Active encryption
    #       Files are being encrypted AND injection activity is still elevated.
    #       Universal signals peak: ldrmodules_not_in_load > 434, vad_exec_count > 5598,
    #       handle_total > 37125, plus encryption artifacts visible.
    #
    #   3 -- Post-encryption
    #       Encryption has completed -- injection anomalies have settled back
    #       to near-benign levels (<80) but encrypted files/notes remain on disk.
    #       The injection/loading phase is over; ransomware may show ransom UI
    #       or have exited. Recovery window -- files are already damaged.
    #
    family = meta.get("family", "")
    # Core universal signals from empirical analysis
    ldr_anomaly      = features.get("ldrmodules_not_in_load", 0)
    vad_exec         = features.get("vad_exec_count", 0)
    handle_total     = features.get("handle_total", 0)
    pslist_count     = features.get("pslist_count", 0)
    malfind_count    = features.get("malfind_count", 0)
    svc_sec_stopped  = features.get("svcscan_security_stopped", 0)
    dlllist_crypto   = features.get("dlllist_crypto_hit_count", 0)
    handle_mutex     = features.get("handle_mutex_count", 0)
    cmd_suspicious   = features.get("cmdline_suspicious_count", 0)
    priv_enabled     = features.get("priv_total_enabled", 0)
    wow64_count      = features.get("pslist_wow64_count", 0)
    ransom_procs     = features.get("pslist_ransom_procs", 0)

    # Encryption artifacts (family-agnostic)
    enc_files        = features.get("filescan_encrypted", 0)
    ransom_notes     = features.get("filescan_ransom_notes", 0)

    # Universal thresholds from empirical peaks (mean across families)
    # These provide cross-family generalization based on analysis
    universal_active = (
        ldr_anomaly > 200 or      # Universal signal: mean peak 434
        vad_exec > 3000 or        # Universal: mean peak 5598
        handle_total > 20000 or   # Universal: mean peak 37125
        pslist_count > 90 or      # Universal: mean peak 95
        malfind_count > 3 or      # Majority: mean peak 6
        svc_sec_stopped > 3 or    # Majority: mean peak 4
        dlllist_crypto > 50 or    # Some families show crypto DLL loading
        handle_mutex > 30 or      # Suspicious handle patterns
        cmd_suspicious > 1        # Suspicious command lines
    )

    universal_preenc = (
        ldr_anomaly > 100 or      # Early elevation in injection
        svc_sec_stopped > 2 or    # Security service manipulation
        wow64_count > 2 or        # 32-bit injection on 64-bit
        priv_enabled > 1000 or    # Privilege escalation
        ransom_procs > 0 or       # Known ransomware processes
        malfind_count > 2         # Injection activity
    )

    # Family-specific adjustments for encryption artifact detection
    #
    # Dharma is a sub-60s encryptor -- by 5s it already has 50+ encrypted files
    # and ldr_anomaly ~60, which the old logic (ldr_threshold_low=70) immediately
    # classified as post-enc (stage 3). Fixes:
    #   - ldr_threshold_low raised to 30 (Dharma fully exits by T090; ldr drops
    #     to near-zero only after process exit, not during active encryption)
    #   - has_enc_artifacts requires a meaningful file count (>500) to distinguish
    #     "just started" (stage 1/2) from "encryption complete" (stage 3)
    #   - enc_files_heavy (>500) used as the post-enc artifact signal
    #   - enc_files_active (>10) combined with still-active injection = stage 2
    #
    if family == "Jigsaw":
        # Jigsaw: screen locker, minimal file encryption, high injection
        has_enc_artifacts = enc_files > 0 or ransom_notes > 0 or malfind_count > 8
        enc_files_active  = enc_files > 0
        ldr_threshold_low = 40  # Lower settlement threshold
    elif family == "Dharma":
        # Dharma: extremely fast encryptor (sub-60s).
        # At T005 ldr~60 + enc_files~60 -- that's stage 1/2, NOT post-enc.
        # Post-enc only when enc count is large AND ldr has fully collapsed.
        has_enc_artifacts = enc_files > 500 or ransom_notes > 0
        enc_files_active  = enc_files > 10   # actively encrypting but not done
        ldr_threshold_low = 30               # ldr only drops this low after process exit
    elif family == "Cerber":
        # Cerber: ldr_anomaly sits at ~60 throughout (never rises above 100),
        # so the default ldr_threshold_low=100 always fires ldr_settled=True.
        # Cerber encrypts file contents in-place -- enc_files stays flat at ~14
        # across all stages, so file count can't gate post-enc either.
        # Key discriminators:
        #   - malfind_count ~11-13 during stages 1-2, drops to ~3 at stage 3
        #   - ldr_anomaly < 20 only at genuine post-enc (T090)
        # Use malfind as the injection-active proxy and ldr < 20 as settled.
        has_enc_artifacts = enc_files > 5 or ransom_notes > 0
        enc_files_active  = enc_files > 5
        ldr_threshold_low = 20               # ldr drops to ~10 only at true post-enc
    else:
        # Default (WannaCry, Benign)
        has_enc_artifacts = enc_files > 3 or ransom_notes > 0
        enc_files_active  = enc_files > 3
        ldr_threshold_low = 100  # Universal signal settles below this

    ldr_active  = ldr_anomaly >= ldr_threshold_low   # injection still running
    ldr_settled = ldr_anomaly < ldr_threshold_low    # injection done / process exited

    # Stage determination
    #
    # Priority order (most specific first):
    #   3 -- post-enc:    large enc artifact count AND injection fully settled
    #   2 -- active enc:  enc artifacts visible AND injection still active
    #   1 -- pre-enc:     ransomware active (injection/handles/privs) but files intact
    #   0 -- benign:      no significant ransomware activity
    #
    if has_enc_artifacts and ldr_settled:
        behavior_stage = 3          # post-encryption: done encrypting, injection gone
    elif (has_enc_artifacts or enc_files_active) and (universal_active or ldr_active):
        behavior_stage = 2          # active encryption: files accumulating + still active
    elif universal_preenc:
        behavior_stage = 1          # pre-enc: ransomware active, files still intact
    else:
        behavior_stage = 0          # benign: no significant activity

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

    # Cache for future runs
    try:
        with open(cache_path, "w") as f:
            json.dump(row, f)
    except Exception:
        pass

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

    print(f"\n[done] Features saved to: {args.out}")
    print(f"[done] Rows: {len(rows)}  |  Feature columns: {len(feat_cols)}")


if __name__ == "__main__":
    main()
