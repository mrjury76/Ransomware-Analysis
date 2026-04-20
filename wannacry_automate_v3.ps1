# =============================================================================
# ransomware_automate_v3.ps1
# Pass -FamilyArg and -NumRunsArg to run non-interactively (e.g. from overnight.ps1)
# Stage-Aware Ransomware Memory Collection - Multi-Family Edition
#
# Changes from v2:
#   - vol3 runs in WSL (not Windows Python) with automatic path translation
#   - Output root moved to D:\ drive
#   - Multi-family config table at the top - pick a family at prompt
#   - VMware snapshot auto-delete after vmem is extracted (saves disk)
#   - Snapshot tracking array so delete-by-name works correctly
#   - WSL autovol4.py replaces per-plugin vol3 calls (PID-filtered output)
# =============================================================================

param(
    [string]$FamilyArg = "",   # pass "ALL" or a family name to skip the prompt
    [int]$NumRunsArg   = 0     # pass a number to skip the cycles prompt
)

# -- SYSTEM CONFIG - edit these once, leave alone after ----------------------
$VMRUN      = "C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"
$VMX        = "C:\Users\Patrick\Documents\Virtual Machines\Windows 10 x64\Windows 10 x64.vmx"
$CLEAN_SNAP = "CleanFamily5"
$VM_USER    = "patrick"
$VM_PASS    = "kali"

# autovol4.py in WSL - full path to the script inside WSL
$AUTOVOL4_WSL = "/home/patrick/tools/volatility3/autovol4.py"

# run_pipeline.py in WSL - set to "" to skip pipeline after collection
$PIPELINE_WSL = "/mnt/c/Users/Patrick/Desktop/MusfiqFinalProject/Ransomware-Analysis/run_pipeline.py"

# Output root on D drive - all families write here
$OUTPUT_ROOT = "D:\Patrick\VMSnapshots\Datasets"

# Snapshot timings - seconds post-launch to capture
# T=0 baseline always captured automatically before launch
$SNAP_OFFSETS = @(5, 10, 20, 45, 90)

# Benign captures only need a single snapshot -- system state doesn't evolve
# over short windows the way ransomware does. One snapshot per rep = 1 data point.
$BENIGN_SNAP_OFFSETS = @(30)

# -- MULTI-FAMILY CONFIG TABLE ------------------------------------------------
# Key   = display name (also used in output folder names and CSV family column)
# Value = path to malware executable INSIDE the guest VM
#         Use "BENIGN" as the value for the benign baseline family.
$FAMILIES = @{
    "WannaCry" = "C:\Malware\WannaCry\WannaCry.exe"
    "Cerber"   = "C:\Malware\Cerber\Cerber.exe"
    "Jigsaw"   = "C:\Malware\Jigsaw\jigsaw"
    # "Petrwrap" = "C:\Malware\Petrwrap\Petrwrap.exe"
    # "Ryuk"     = "C:\Malware\Ryuk\Ryuk.exe"
    "Dharma"   = "C:\Malware\Dharma\Dharma\Dharma.exe"
    "Benign"   = "BENIGN"
}
# Add/remove entries above to match what you actually have staged in the VM.
# The guest path must be the full Windows path inside the guest.

# -- BENIGN PROCESS POOL -----------------------------------------------------
# Common user applications to simulate normal desktop activity.
# Each entry: [display name, guest exe path, optional arguments]
# The script randomly picks $BENIGN_PICK_COUNT of these per run.
$BENIGN_PICK_COUNT = 5

$BENIGN_PROCESSES = @(
    @{ Name = "Notepad";       Exe = "C:\Windows\System32\notepad.exe";                Args = "" }
    @{ Name = "WordPad";       Exe = "C:\Program Files\Windows NT\Accessories\wordpad.exe"; Args = "" }
    @{ Name = "Calculator";    Exe = "C:\Windows\System32\calc.exe";                   Args = "" }
    @{ Name = "Paint";         Exe = "C:\Windows\System32\mspaint.exe";                Args = "" }
    @{ Name = "Explorer";      Exe = "C:\Windows\explorer.exe";                        Args = "C:\Users\$VM_USER\Documents" }
    @{ Name = "PowerShell";    Exe = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"; Args = "-NoExit -Command Get-Process" }
    @{ Name = "CMD";           Exe = "C:\Windows\System32\cmd.exe";                    Args = "/k dir C:\Users\$VM_USER" }
    @{ Name = "SnippingTool";  Exe = "C:\Windows\System32\SnippingTool.exe";           Args = "" }
    @{ Name = "Firefox";       Exe = "C:\Program Files\Mozilla Firefox\firefox.exe";   Args = "" }
    @{ Name = "Chrome";        Exe = "C:\Program Files\Google\Chrome\Application\chrome.exe"; Args = "" }
    @{ Name = "Word";          Exe = "C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"; Args = "" }
    @{ Name = "Excel";         Exe = "C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE";   Args = "" }
    # @{ Name = "Outlook";       Exe = "C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE"; Args = "" }
    @{ Name = "MediaPlayer";   Exe = "C:\Program Files\Windows Media Player\wmplayer.exe"; Args = "" }
    @{ Name = "TaskManager";   Exe = "C:\Windows\System32\Taskmgr.exe";               Args = "" }
)

# -- FUNCTION DEFINITIONS -----------------------------------------------------
function Log {
    param([string]$msg, [string]$color = "White")
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $logFile -Value $line
}

function Run-VMRun {
    param([string[]]$ArgList)
    $output = & $VMRUN @ArgList 2>&1
    $rc = $LASTEXITCODE
    if ($rc -ne 0 -and $output) {
        Log "  [vmrun] $($ArgList[0]) error: $($output -join ' ')" DarkGray
    }
    return $rc
}

# -- RUNTIME PROMPT (bypassed if -FamilyArg / -NumRunsArg passed) ------------
if ($FamilyArg -eq "" -or $NumRunsArg -eq 0) {
    Write-Host ""
    Write-Host "Available families:" -ForegroundColor Cyan
    $FAMILIES.Keys | Sort-Object | ForEach-Object { Write-Host "  - $_" }
    Write-Host "  - ALL  (run every family in sequence)" -ForegroundColor Cyan
    Write-Host ""
}

$familyInput = if ($FamilyArg -ne "") { $FamilyArg } else { (Read-Host "Which family to run? (name or ALL)").Trim() }
$NUM_RUNS    = if ($NumRunsArg -gt 0)  { $NumRunsArg } else { [int](Read-Host "How many full infection cycles per family?") }
$runAnalysis = $true

if ($familyInput -eq "ALL") {
    $familiesToRun = @($FAMILIES.Keys | Sort-Object)
} elseif ($FAMILIES.ContainsKey($familyInput)) {
    $familiesToRun = @($familyInput)
} else {
    Write-Host "[!] Unknown family '$familyInput'. Check the FAMILIES table." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Families queued: $($familiesToRun -join ', ')" -ForegroundColor Cyan
Write-Host "Cycles per family: $NUM_RUNS" -ForegroundColor Cyan
Write-Host ""

# -- PATH TRANSLATION: Windows -> WSL -----------------------------------------
# autovol4.py runs inside WSL so it needs Linux-style paths.
# D:\foo\bar.vmem -> /mnt/d/foo/bar.vmem
function ConvertTo-WslPath {
    param([string]$WinPath)
    $p = $WinPath.Replace("\", "/")
    if ($p -match "^([A-Za-z]):/(.+)$") {
        return "/mnt/$($Matches[1].ToLower())/$($Matches[2])"
    }
    return $p
}

# -- AUTOVOL4 RUNNER (WSL) ----------------------------------------------------
# Single call to autovol4.py - runs all plugins, filters by malware PID tree,
# writes vol3_combined.csv + per-plugin raw CSVs into $OutWinDir.
function Invoke-AutoVol4 {
    param(
        [string]$VmemWinPath,
        [string]$OutWinDir
    )

    if (-not (Test-Path $VmemWinPath)) {
        Log "  [AUTOVOL4] vmem not found: $VmemWinPath" Red
        return
    }

    New-Item -ItemType Directory -Force -Path $OutWinDir | Out-Null

    $wslVmem   = ConvertTo-WslPath $VmemWinPath
    $wslOutDir = ConvertTo-WslPath $OutWinDir
    $wslCmd    = "python3 '$AUTOVOL4_WSL' --family '$FAMILY' --vmem '$wslVmem' --output-dir '$wslOutDir'"

    Log "  [AUTOVOL4] Starting ($FAMILY)..." White

    $startArgs = @{
        FilePath     = "wsl.exe"
        ArgumentList = @("bash", "-c", $wslCmd)
        NoNewWindow  = $true
        PassThru     = $true
        Wait         = $true
    }
    $proc = Start-Process @startArgs

    if ($proc.ExitCode -eq 0) {
        Log "  [AUTOVOL4] Done -> $OutWinDir\vol3_combined.csv" Green
    } else {
        Log "  [AUTOVOL4] FAILED (exit $($proc.ExitCode)) - check $OutWinDir\run.log" Red
    }
}

# -- VMEM EXTRACTION ----------------------------------------------------------
# VMware Workstation writes snapshot memory to the same folder as the .vmx.
# We grab the most recently written .vmem to be safe.
function Extract-Vmem {
    param([string]$SnapName, [string]$DestDir)

    $vmDir     = Split-Path $VMX -Parent
    $vmemFiles = Get-ChildItem -Path $vmDir -Filter "*.vmem" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending

    if ($vmemFiles.Count -eq 0) {
        Log "  [VMEM] No .vmem files found in $vmDir" Red
        return $null
    }

    $src      = $vmemFiles[0].FullName
    $destPath = Join-Path $DestDir "${SnapName}.vmem"

    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Log "  [VMEM] $($vmemFiles[0].Name)  ->  $destPath" Yellow
    Copy-Item -Path $src -Destination $destPath -Force

    if (Test-Path $destPath) {
        $mb = [math]::Round((Get-Item $destPath).Length / 1MB, 1)
        Log "  [VMEM] Copied ${mb} MB" Green
        return $destPath
    }

    Log "  [VMEM] Copy failed" Red
    return $null
}

# -- SNAPSHOT DELETE HELPER ---------------------------------------------------
function Remove-VmSnap {
    param([string]$SnapName)
    Log "  [SNAP] Deleting VMware snapshot: $SnapName" DarkGray
    $rc = Run-VMRun @("deleteSnapshot", $VMX, $SnapName)
    if ($rc -eq 0) {
        Log "  [SNAP] Deleted $SnapName" DarkGray
    } else {
        Log "  [SNAP] Could not delete $SnapName (exit $rc) - delete manually" Yellow
    }
}

# -- RUN EACH FAMILY ----------------------------------------------------------
foreach ($FAMILY in $familiesToRun) {

$MALWARE_PATH = $FAMILIES[$FAMILY]
$timestamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$SESSION_DIR  = Join-Path $OUTPUT_ROOT "${FAMILY}_${timestamp}"
New-Item -ItemType Directory -Force -Path $SESSION_DIR | Out-Null
$logFile = Join-Path $SESSION_DIR "session.log"

$activeOffsets = if ($MALWARE_PATH -eq "BENIGN") { $BENIGN_SNAP_OFFSETS } else { $SNAP_OFFSETS }
$totalRuns     = $NUM_RUNS * $activeOffsets.Count

Log "============================================" Cyan
Log " $FAMILY Collection Session"               Cyan
Log " Guest path:  $MALWARE_PATH"               Cyan
Log " Runs/offset: $NUM_RUNS"                   Cyan
Log " Offsets:     $($SNAP_OFFSETS -join 's, ')s" Cyan
Log " Total runs:  $totalRuns"                  Cyan
Log " Output:      $SESSION_DIR"                Cyan
Log "============================================" Cyan

# -- MAIN LOOP ----------------------------------------------------------------
# Each run: fresh infection, ONE snapshot at the target offset, done.
# Outer loop = reps, inner loop = offsets so we cycle T+15,30,60... before repeating.
$runIndex = 0
for ($rep = 1; $rep -le $NUM_RUNS; $rep++) {
    foreach ($targetOffset in $activeOffsets) {
        $runIndex++

        Log "" White
        Log "==========================================" Cyan
        Log " RUN $runIndex / $totalRuns - $FAMILY @ T+${targetOffset}s (rep $rep)" Cyan
        Log "==========================================" Cyan

        $runDir = Join-Path $SESSION_DIR "T$('{0:D3}' -f $targetOffset)_rep$('{0:D2}' -f $rep)"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null

        # -- Revert to clean --------------------------------------------------
        Log "[+] Stopping VM..." DarkGray
        Run-VMRun @("stop", $VMX, "hard") | Out-Null
        Start-Sleep -Seconds 2

        Log "[+] Reverting to $CLEAN_SNAP..." Yellow
        $rc = Run-VMRun @("revertToSnapshot", $VMX, $CLEAN_SNAP)
        if ($rc -ne 0) { Log "[!] Revert failed - skipping" Red; continue }
        Log "    -> Reverted" Green

        # -- Start VM and wait for boot ---------------------------------------
        Log "[+] Starting VM..." Yellow
        $rc = Run-VMRun @("start", $VMX)
        if ($rc -ne 0) { Log "[!] Start failed - skipping" Red; continue }

        Log "[+] Waiting 20s for boot..." White
        Start-Sleep -Seconds 20

        # -- Launch processes -------------------------------------------------
        $launchTime = Get-Date

        if ($MALWARE_PATH -eq "BENIGN") {
            # Pick random benign user processes for this run
            $picked = $BENIGN_PROCESSES | Get-Random -Count ([math]::Min($BENIGN_PICK_COUNT, $BENIGN_PROCESSES.Count))
            $pickedNames = ($picked | ForEach-Object { $_.Name }) -join ", "
            Log "[+] Launching $($picked.Count) benign processes: $pickedNames" Yellow

            $launchFailed = $false
            foreach ($proc in $picked) {
                $runArgs = @("-gu", $VM_USER, "-gp", $VM_PASS,
                             "runProgramInGuest", $VMX,
                             "-activeWindow", "-nowait",
                             $proc.Exe)
                if ($proc.Args -ne "") { $runArgs += $proc.Args }

                $rc = Run-VMRun $runArgs
                if ($rc -ne 0) {
                    Log "    [!] $($proc.Name) failed to launch (exit $rc) - skipping it" Yellow
                } else {
                    Log "    -> $($proc.Name) launched" Green
                }
                Start-Sleep -Milliseconds 15000
            }

            # Write which processes were picked so autovol4/pipeline can reference them
            $picked | ForEach-Object { $_.Name } | Set-Content (Join-Path $runDir "benign_processes.txt")
            Log "    -> Benign process list saved to benign_processes.txt" Green
        } else {
            Log "[+] Launching $FAMILY..." Yellow

            $rc = Run-VMRun @("-gu", $VM_USER, "-gp", $VM_PASS,
                              "runProgramInGuest", $VMX,
                              "-activeWindow", "-nowait",
                              $MALWARE_PATH)

            if ($rc -ne 0) { Log "[!] Launch failed - skipping" Red; continue }
            Log "    -> $FAMILY launched" Green
        }

        # -- Wait for target offset then snapshot -----------------------------
        $alreadyElapsed = [int]((Get-Date) - $launchTime).TotalSeconds
        $sleepFor       = [math]::Max(0, $targetOffset - $alreadyElapsed)

        if ($sleepFor -gt 0) {
            Log "[+] Sleeping ${sleepFor}s (target T+${targetOffset}s)..." White
            Start-Sleep -Seconds $sleepFor
        }

        $actualOffset = [int]((Get-Date) - $launchTime).TotalSeconds
        $snapTs       = Get-Date -Format "HHmmss"
        $snapName     = "${FAMILY}_T$('{0:D3}' -f $targetOffset)_rep$('{0:D2}' -f $rep)_${snapTs}"

        Log "[+] Snapshot at T+${actualOffset}s: $snapName" Yellow
        $rc = Run-VMRun @("snapshot", $VMX, $snapName)
        if ($rc -ne 0) { Log "    [!] Snapshot failed" Red; continue }
        Log "    -> Taken" Green

        # -- Wait for VMware to flush vmem ------------------------------------
        Log "[+] Waiting 30s for VMware to flush vmem..." White
        Start-Sleep -Seconds 30

        # -- Copy vmem --------------------------------------------------------
        $vmDir   = Split-Path $VMX -Parent
        $vmemSrc = (Get-ChildItem -Path $vmDir -Filter "*.vmem" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

        $destPath = Join-Path $runDir "${snapName}.vmem"

        @{
            family          = $FAMILY
            run             = $runIndex
            target_offset_s = $targetOffset
            actual_offset_s = $actualOffset
            rep             = $rep
            snap_name       = $snapName
            launch_time     = $launchTime.ToString("o")
            snap_time       = (Get-Date).ToString("o")
            stage_hint      = if ($MALWARE_PATH -eq "BENIGN") { 0 }
                              elseif ($actualOffset -lt 20)  { 1 }
                              elseif ($actualOffset -lt 50)  { 2 }
                              elseif ($actualOffset -lt 150) { 3 }
                              else                           { 4 }
        } | ConvertTo-Json | Set-Content (Join-Path $runDir "meta.json")

        if ($vmemSrc -and (Test-Path $vmemSrc)) {
            Log "  [VMEM] Copying -> $destPath" Yellow
            Copy-Item -Path $vmemSrc -Destination $destPath -Force
            $mb = [math]::Round((Get-Item $destPath).Length / 1MB, 1)
            Log "  [VMEM] Copied ${mb} MB" Green

            if ($runAnalysis) {
                Log "[+] autovol4 $snapName..." Yellow
                Invoke-AutoVol4 -VmemWinPath $destPath -OutWinDir $runDir
            }
        } else {
            Log "  [VMEM] Source not found" Red
        }

        # -- Delete VMware snapshot -------------------------------------------
        Run-VMRun @("deleteSnapshot", $VMX, $snapName) | Out-Null
        Log "[+] Run $runIndex complete." Green
    }
}

# -- Final revert -------------------------------------------------------------
Log "" White
Log "[+] All cycles done. Reverting to clean snapshot..." Yellow
Run-VMRun @("stop", $VMX, "hard") | Out-Null
Start-Sleep -Seconds 2
$rc = Run-VMRun @("revertToSnapshot", $VMX, $CLEAN_SNAP)
if ($rc -eq 0) { Log "    -> Reverted to $CLEAN_SNAP" Green } else { Log "    [!] Final revert failed" Red }

Log "" White
Log "============================================" Cyan
Log " Session complete." Cyan
Log " Family:  $FAMILY" Cyan
Log " Output:  $SESSION_DIR" Cyan
Log "============================================" Cyan

} # end foreach family

Write-Host ""
Write-Host "All families done." -ForegroundColor Green

# -- Run pipeline (autovol + features + training) in WSL ---------------------
if ($PIPELINE_WSL -ne "") {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host " Running analysis pipeline in WSL..."       -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan

    $wslScanDir = ConvertTo-WslPath $OUTPUT_ROOT
    $wslCmd     = "python3 $PIPELINE_WSL --scan-dir $wslScanDir"

    $proc = Start-Process -FilePath "wsl.exe" `
                          -ArgumentList @("bash", "-c", $wslCmd) `
                          -NoNewWindow -PassThru -Wait

    if ($proc.ExitCode -eq 0) {
        Write-Host "Pipeline complete." -ForegroundColor Green
    } else {
        Write-Host "[!] Pipeline failed (exit $($proc.ExitCode)) - run manually in WSL." -ForegroundColor Red
    }
}
