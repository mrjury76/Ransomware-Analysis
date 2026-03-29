# =============================================================================
# ransomware_automate_v3.ps1
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

# -- SYSTEM CONFIG - edit these once, leave alone after ----------------------
$VMRUN      = "C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"
$VMX        = "C:\Users\Patrick\Documents\Virtual Machines\Windows 10 x64\Windows 10 x64.vmx"
$CLEAN_SNAP = "CleanFamily3"
$VM_USER    = "patrick"
$VM_PASS    = "kali"

# autovol4.py in WSL - full path to the script inside WSL
$AUTOVOL4_WSL = "/home/patrick/tools/volatility3/autovol4.py"

# Output root on D drive - all families write here
$OUTPUT_ROOT = "D:\Patrick\VMSnapshots"

# Snapshot timings - seconds post-launch to capture
# T=0 baseline always captured automatically before launch
$SNAP_OFFSETS = @(15, 30, 60, 120, 180, 240)

# -- MULTI-FAMILY CONFIG TABLE ------------------------------------------------
# Key   = display name (also used in output folder names and CSV family column)
# Value = path to malware executable INSIDE the guest VM
$FAMILIES = @{
    "WannaCry" = "C:\Malware\WannaCry\WannaCry.exe"
    "Cerber"   = "C:\Malware\Cerber\Cerber.exe"
    "Jigsaw"   = "C:\Malware\Jigsaw\jigsaw"
    "Petrwrap" = "C:\Malware\Petrwrap\Petrwrap.exe"
}
# Add/remove entries above to match what you actually have staged in the VM.
# The guest path must be the full Windows path inside the guest.

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

# -- RUNTIME PROMPT -----------------------------------------------------------
Write-Host ""
Write-Host "Available families:" -ForegroundColor Cyan
$FAMILIES.Keys | Sort-Object | ForEach-Object { Write-Host "  - $_" }
Write-Host "  - ALL  (run every family in sequence)" -ForegroundColor Cyan
Write-Host ""

$familyInput = (Read-Host "Which family to run? (name or ALL)").Trim()
$NUM_RUNS    = [int](Read-Host "How many full infection cycles per family?")
$runAnalysis = $false   # set to $true to run autovol4 on each snapshot after collection

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
        ArgumentList = @("-e", "bash", "-c", $wslCmd)
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

$totalRuns = $NUM_RUNS * $SNAP_OFFSETS.Count

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
    foreach ($targetOffset in $SNAP_OFFSETS) {
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

        # -- Launch malware ---------------------------------------------------
        $launchTime = Get-Date
        Log "[+] Launching $FAMILY..." Yellow

        $rc = Run-VMRun @("-gu", $VM_USER, "-gp", $VM_PASS,
                          "runProgramInGuest", $VMX,
                          "-activeWindow", "-nowait",
                          $MALWARE_PATH)

        if ($rc -ne 0) { Log "[!] Launch failed - skipping" Red; continue }
        Log "    -> $FAMILY launched" Green

        # -- Wait for target offset then snapshot -----------------------------
        $alreadyElapsed = [int]((Get-Date) - $launchTime).TotalSeconds
        $sleepFor       = [math]::Max(0, $targetOffset - $alreadyElapsed)

        if ($sleepFor -gt 0) {
            Log "[+] Sleeping ${sleepFor}s (target T+${targetOffset}s)..." White
            Start-Sleep -Seconds $sleepFor
        }

        $actualOffset = [int]((Get-Date) - $launchTime).TotalSeconds
        $snapName     = "${FAMILY}_T$('{0:D3}' -f $targetOffset)_rep$('{0:D2}' -f $rep)"

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
            stage_hint      = if ($actualOffset -lt 20)  { 0 }
                              elseif ($actualOffset -lt 50)  { 1 }
                              elseif ($actualOffset -lt 150) { 2 }
                              else                           { 3 }
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
