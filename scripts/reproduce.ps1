# Reproduce every result of the paper from the released weights (and, with -Train, retrain them first).
# Run from the repository root after preparing the data (README, "Data"):
#   powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Train       # retrain the 18 estimators first
#   powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Python "C:\path\to\python.exe"
# Each step is skipped when its output already exists; delete an output to recompute it. The released
# per-image CSVs are such outputs, so by default only the analyses are re-run from them. Use -Rescore to
# re-score the estimators on the images (GPU recommended; about 1-2 hours in total).

param([string]$Python = "python", [switch]$Train, [switch]$Rescore)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)
$cal = "evaluation_outputs/calibre"
$seeds = @(42, 123, 7)

function Run([string]$Name, [string]$Done, [string[]]$Arguments) {
    if ($Done -and (Test-Path $Done)) { Write-Host "[skip] $Name ($Done exists)"; return }
    Write-Host "`n==== $Name  $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)." }
}

function Score([string]$Output, [string[]]$Extra) {
    $csv = "$cal/$([IO.Path]::GetFileNameWithoutExtension($Output))_per_image.csv"
    if ($Rescore -and (Test-Path $csv)) { Remove-Item $csv }
    Run "score $Output" $csv (@("-m", "src.test_sge_temperature", "--temperatures", "1", "--output", $Output) + $Extra)
}

# 1. Topology targets for the DRIVE training images (needed by the training loader and the validation split).
Run "topology targets" "datasets_topology/DRIVE_train/dataset_summary.json" @(
    "-m", "src.generate_topology_targets", "--config", "configs/data.yaml", "--dataset", "DRIVE_train")

# 2. Training (optional; the weights are in checkpoints/).
if ($Train) {
    foreach ($s in $seeds) {
        foreach ($v in @("R2", "P1", "R2ns", "R2w", "P1w", "SEU")) {
            $out = "checkpoints/sge_$($v)_seed$s"
            if (Test-Path "$out/sge.pth") { Remove-Item "$out/sge.pth" }
            $factors = if ($v -in @("R2w", "P1w")) { @("--factors", "0.75", "1", "2", "4", "6", "8") } else { @() }
            Run "train $v seed $s" "" (@("-m", "src.pretrain_sge", "--config", "configs/sge_$($v)_seed$s.yaml", "--epochs", "5") + $factors + @("--output", $out))
        }
    }
}

# 3. Calibre scores (per-image CSVs).
$ck = { param($v, $s) "checkpoints/sge_$($v)_seed$s/sge.pth" }
$all5 = @("--cases", "DRIVE_test", "STARE", "CHASE_DB1", "HRF", "FIVES_test")
Score "sge_R2_standalone.md" @("--estimators", "R2=$(& $ck R2 42)", "P1=$(& $ck P1 42)")
foreach ($s in @(123, 7)) { Score "sge_standalone_s$s.md" @("--estimators", "R2=$(& $ck R2 $s)", "P1=$(& $ck P1 $s)") }
$est = foreach ($s in $seeds) { "R2_s$s=$(& $ck R2 $s)"; "P1_s$s=$(& $ck P1 $s)" }
Score "sge_drive_nativegrid.md" (@("--cases", "DRIVE_test", "--native-grid", "--estimators") + $est)
foreach ($s in $seeds) {
    Score "sge_fives_s$s.md" @("--cases", "FIVES_test", "--native-grid", "--estimators", "R2_s$s=$(& $ck R2 $s)", "P1_s$s=$(& $ck P1 $s)")
    Score "sge_R2ns_s$s.md" @("--native-grid", "--estimators", "R2ns_s$s=$(& $ck R2ns $s)")
    Score "sge_wide_s$s.md" (@("--native-grid") + $all5 + @("--estimators", "R2w_s$s=$(& $ck R2w $s)", "P1w_s$s=$(& $ck P1w $s)"))
    Score "sge_SEU_s$s.md" (@("--native-grid") + $all5 + @("--estimators", "SEU_s$s=$(& $ck SEU $s)"))
}

# 4. Analyses. Steps marked (probability maps) need evaluation_outputs/csr_probs (README, "Mask-width baseline").
$probs = Test-Path "evaluation_outputs/csr_probs"
Run "Table 3: R2 vs P1" "" @("-m", "src.r2_vs_p1")
Run "Table 5: weight sharing" "" @("-m", "src.sharing_ablation_analysis")
Run "Table 6, S3: wide-scale training" "" @("-m", "src.wide_scale_analysis")
if ($probs) {
    Run "Tables S5-S6: mask width, test-set threshold (probability maps)" "" @("-m", "src.mask_calibre_baseline")
    Run "Table 4, S4: mask width, validation threshold (probability maps)" "" @("-m", "src.mask_threshold_validation")
    Run "Table 6: FIVES mask width (probability maps)" "" @("-m", "src.fives_calibre_analysis")
} else {
    Write-Host "[skip] mask-width analyses: evaluation_outputs/csr_probs not found"
    Run "Table 4, S4: mask width, validation threshold (saved scores)" "" @("-m", "src.mask_threshold_validation", "--analyse-only")
}
$obs = @("-m", "src.second_observer"); if (-not $probs -or -not $Rescore) { $obs += "--analyse-only" }
Run "Table 9: second observer" "" $obs
if ($Rescore) { Run "second observer, SEU" "" @("-m", "src.second_observer", "--models", "SEU") }
$syn = @("-m", "src.synthetic_calibre"); if (-not $Rescore) { $syn += "--analyse-only" }
Run "Tables 7-8: synthetic vessels" "" $syn
if ($Rescore) { Run "synthetic vessels, SEU" "" @("-m", "src.synthetic_calibre", "--models", "SEU") }
$lss = @("-m", "src.scale_selection_baseline"); if (-not $Rescore) { $lss += "--analyse-only" }
Run "Table 10: classical scale selection" "" $lss
Run "Table 11: SEU" "" @("-m", "src.seu_baseline_analysis")
Run "Table S7: equivariance checks" "" @("-m", "src.seu_checks")
Run "Figure 1" "" @("-m", "src.fig_architecture")
Run "Supplementary tables" "" @("scripts/make_supplement.py")
Write-Host "`nDone $(Get-Date -Format 'yyyy-MM-dd HH:mm'). Results: $cal"
