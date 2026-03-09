# ===== CONFIG =====
$VMRUN      = "C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"
$VMX        = "C:\Users\Patrick\Documents\Virtual Machines\Windows 10 x64\Windows 10 x64.vmx"
$CLEAN_SNAP = "CLEANSnapshot8"
$VM_USER    = "patrick"
$VM_PASS    = "kali"
$WANNACRY   = "C:\Malware\WannaCry\WannaCry.exe"
$SNAP_BASE  = 18
# ==================

$NUM_SNAPSHOTS = [int](Read-Host "How many snapshots do you want to take?")
$INTERVAL      = [int](Read-Host "How many seconds to wait before taking each snapshot?")

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " WannaCry Automation Script"
Write-Host " Snapshots:  $NUM_SNAPSHOTS"
Write-Host " Interval:   $INTERVAL seconds each"
Write-Host " Clean snap: $CLEAN_SNAP"
Write-Host "============================================" -ForegroundColor Cyan

for ($i = 1; $i -le $NUM_SNAPSHOTS; $i++) {
    $SNAP_NUM  = $SNAP_BASE + $i - 1
    $SNAP_NAME = "V$SNAP_NUM"

    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
    Write-Host " Run $i of $NUM_SNAPSHOTS  ->  $SNAP_NAME" -ForegroundColor Cyan
    Write-Host "--------------------------------------------" -ForegroundColor Cyan

    # ── Revert to clean snapshot ──────────────────────────────────────────────
    Write-Host "[+] Reverting to $CLEAN_SNAP..." -ForegroundColor Yellow
    & $VMRUN revertToSnapshot $VMX $CLEAN_SNAP
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to revert. Skipping run $i." -ForegroundColor Red
        continue
    }
    Write-Host "     ->Reverted successfully"

    # ── Start VM ──────────────────────────────────────────────────────────────
    Write-Host "[+] Starting VM..." -ForegroundColor Yellow
    & $VMRUN start $VMX
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to start VM. Skipping run $i." -ForegroundColor Red
        continue
    }

    # Write-Host "Turning off Windows Defender real-time monitoring..." -ForegroundColor Yellow
    # & $VMRUN -gu $VM_USER -gp $VM_PASS runProgramInGuest $VMX -activeWindow "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" "-Command Set-MpPreference -DisableRealtimeMonitoring `$true"
    # Start-Sleep -Seconds 5

    Write-Host "[+] Waiting 20 seconds for VM to boot..."
    Start-Sleep -Seconds 20

    # ── Execute WannaCry ──────────────────────────────────────────────────────
    Write-Host "[+] Executing WannaCry..." -ForegroundColor Yellow
    & $VMRUN -gu $VM_USER -gp $VM_PASS runProgramInGuest $VMX -activeWindow -nowait $WANNACRY
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to execute WannaCry. Skipping run $i." -ForegroundColor Red
        continue
    }
    Write-Host "     ->WannaCry launched"

    # ── Wait then snapshot ────────────────────────────────────────────────────
    Write-Host "[+] Waiting $INTERVAL seconds before snapshot..." -ForegroundColor Yellow
    Start-Sleep -Seconds $INTERVAL

    Write-Host "[+] Taking snapshot: $SNAP_NAME..." -ForegroundColor Yellow
    & $VMRUN snapshot $VMX $SNAP_NAME
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to take snapshot $SNAP_NAME" -ForegroundColor Red
    } else {
        Write-Host "    -> Snapshot $SNAP_NAME saved" -ForegroundColor Green
    }
}

# ── Final revert to clean ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[+] All runs complete. Reverting to clean snapshot..." -ForegroundColor Yellow
& $VMRUN revertToSnapshot $VMX $CLEAN_SNAP
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Failed to revert to clean snapshot." -ForegroundColor Red
} else {
    Write-Host "    -> Reverted to $CLEAN_SNAP" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Done."
Write-Host " Snapshots saved: Snapshot$SNAP_BASE to Snapshot$($SNAP_BASE + $NUM_SNAPSHOTS - 1)"
Write-Host " Run your Volatility script against each snapshot."
Write-Host "============================================" -ForegroundColor Cyan