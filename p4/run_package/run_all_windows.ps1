# P4 full run: environment report, smoke test of every stage, full run of every stage, zip of the results.
#   powershell -ExecutionPolicy Bypass -File .\run_all_windows.ps1
# Uses the Python environment C:\p2\.venv and the model cache C:\p2\hf_cache (the same as P2 and P3).
# Running it again continues where it stopped: finished stages return at once.
# A failing stage never stops the stages after it. Every stage writes results\logs\<stage>.log .
# Rehearsal without models (README_RU.md, step 2; a few minutes; writes results\mock, adapters\mock and p4_results_mock_*.zip):
#   powershell -ExecutionPolicy Bypass -File .\run_all_windows.ps1 -Backend mock      (on another machine add -Python <python.exe>)
# This file is plain ASCII on purpose: Windows PowerShell 5.1 reads a script without BOM in the local code page.
param(
  [string]$Backend = "hf",
  [string]$Python = ""
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

# ---------------------------------------------------------------------------------------------------------------------
# Python and environment
# ---------------------------------------------------------------------------------------------------------------------
$p2 = "C:\p2"
$py = ""
if ($Python -ne "") {
  $py = $Python
} elseif (Test-Path -LiteralPath (Join-Path $p2 ".venv\Scripts\python.exe")) {
  $py = Join-Path $p2 ".venv\Scripts\python.exe"
  $env:HF_HOME = Join-Path $p2 "hf_cache"
}
if ($py -eq "") {
  Write-Host "ERROR: the Python environment C:\p2\.venv\Scripts\python.exe was not found. Nothing was run."
  Write-Host "       See README_RU.md, section 5 (if something goes wrong)."
  exit 1
}
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
Remove-Item Env:\P4_DATA_DIR -ErrorAction SilentlyContinue
Remove-Item Env:\P4_OUT_DIR -ErrorAction SilentlyContinue
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }

$resBase = Join-Path $PSScriptRoot "results"
$adp = "adapters"
if ($Backend -ne "hf") {
  $resBase = Join-Path $resBase $Backend
  $adp = "adapters/" + $Backend
}
$logDir = Join-Path $resBase "logs"
$smokeDir = Join-Path $resBase "smoke"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $smokeDir | Out-Null
$smokeFile = Join-Path $smokeDir "status.json"
$fullFile = Join-Path $resBase "status.json"

$transcriptOn = $false
try {
  Start-Transcript -Path (Join-Path $resBase ("transcript_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")) -Append | Out-Null
  $transcriptOn = $true
} catch {
  Write-Host ("transcript not started: " + $_.Exception.Message)
}

# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------
$script:stageOk = $false

function Save-Json {
  param($Object, [string]$Path)
  try {
    $json = ConvertTo-Json -InputObject $Object -Depth 4
    [System.IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding $false))
  } catch {
    Write-Host ("could not write " + $Path + " : " + $_.Exception.Message)
  }
}

# Runs one Python stage. The result is left in $script:stageOk (true = exit code 0).
# The function returns nothing on purpose: the output of python.exe goes straight to the console.
function Invoke-Stage {
  param([string]$Name, [string[]]$PyArgs)
  $log = Join-Path $logDir ($Name + ".log")
  $env:P4_LOG_FILE = $log
  $env:P4_STAGE = $Name
  Write-Host ""
  Write-Host ("=== [" + (Get-Date -Format "HH:mm:ss") + "] START " + $Name)
  $t0 = Get-Date
  $code = -1
  try {
    $global:LASTEXITCODE = -1
    & $py @PyArgs
    $code = $LASTEXITCODE
  } catch {
    $code = -1
    $msg = "STAGE " + $Name + " could not be started: " + $_.Exception.Message
    Write-Host $msg
    try { Add-Content -LiteralPath $log -Value $msg } catch { }
  }
  $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
  $script:stageOk = ($code -eq 0)
  $word = "FAILED"
  if ($script:stageOk) { $word = "OK" }
  Write-Host ("=== [" + (Get-Date -Format "HH:mm:ss") + "] END   " + $Name + " : " + $word + " (exit code " + $code + ", " + $min + " min)")
  if (-not $script:stageOk) { Write-Host ("    the error is in " + $log + " ; the next stages still run") }
}

# Runs one stage of the full run if its gate is open, and records "ok" / "failed" / the skip word in $full.
function Invoke-FullStage {
  param([string]$Key, [bool]$Gate, [string]$SkipWord, [string]$Name, [string[]]$PyArgs)
  if (-not $Gate) {
    Write-Host ""
    Write-Host ("=== SKIPPED " + $Name + " : " + $SkipWord)
    $full[$Key] = $SkipWord
  } else {
    Invoke-Stage -Name $Name -PyArgs $PyArgs
    if ($script:stageOk) { $full[$Key] = "ok" } else { $full[$Key] = "failed" }
  }
  Save-Json -Object $full -Path $fullFile
}

Write-Host ("python: " + $py + " ; HF_HOME: " + $env:HF_HOME + " ; backend: " + $Backend)
if ($Backend -eq "hf") {
  Write-Host "Do not close this window. Expected total: 7 to 10 hours. The plan with expected times is printed below."
} else {
  Write-Host ("TEST MODE (" + $Backend + "): no real model is loaded and this takes a few minutes. The times printed below are those of the real run.")
}

# The full runs need P4_COMMIT.txt (first line = 40-character hash of the pre-registration commit). Without it only the
# environment report and the smoke tests run; the full stages are skipped and the summary says so.
$commitOk = $true
if ($Backend -eq "hf") {
  $commitOk = $false
  $commitFile = Join-Path $PSScriptRoot "P4_COMMIT.txt"
  if (Test-Path -LiteralPath $commitFile) {
    $firstLine = [string](Get-Content -LiteralPath $commitFile -TotalCount 1)
    if ($firstLine -cmatch "^[0-9a-f]{40}(\s|\z)") { $commitOk = $true }
  }
  if (-not $commitOk) {
    Write-Host ""
    Write-Host "ERROR: P4_COMMIT.txt is missing or its first line is not a 40-character commit hash."
    Write-Host "       Only the smoke tests will run. Unpack the package again into a new folder (README_RU.md, section 5)."
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# 0. Environment report (prints the plan with the expected time of every stage)
# ---------------------------------------------------------------------------------------------------------------------
Invoke-Stage -Name "00_env_report" -PyArgs @("scripts/env_report.py", "--backend", $Backend)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Smoke test of every stage (about 12 minutes in total)
# ---------------------------------------------------------------------------------------------------------------------
$smoke = [ordered]@{ infer_8b = $false; infer_4b = $false; sft = $false; sft_infer = $false; grpo = $false; prep = "lean"; adapter = "" }
$smokeOut = $adp + "/smoke"
$smokeOutKbit = $adp + "/smoke_kbit"

Invoke-Stage -Name "s1_smoke_infer_8b" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_8b", "--items", "data/smoke_test.jsonl", "--variants", "V1,V2", "--limit", "3", "--backend", $Backend)
$smoke["infer_8b"] = $script:stageOk
Save-Json -Object $smoke -Path $smokeFile

Invoke-Stage -Name "s2_smoke_infer_4b" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_4b_2507", "--items", "data/smoke_test.jsonl", "--variants", "V1,V2", "--limit", "3", "--backend", $Backend)
$smoke["infer_4b"] = $script:stageOk
Save-Json -Object $smoke -Path $smokeFile

# Fine-tuning smoke: a few optimizer steps, one checkpoint, and a memory test on the longest row of the real training file.
$prep = "lean"
$smokeAdapter = $smokeOut
Invoke-Stage -Name "s3_smoke_sft" -PyArgs @("scripts/train_sft.py", "--seed", "0", "--train", "data/smoke_train.jsonl", "--limit", "8", "--max_steps", "4", "--ckpt_every", "2", "--probe_file", "data/train.jsonl", "--out", $smokeOut, "--prep", "lean", "--backend", $Backend)
if (-not $script:stageOk) {
  Write-Host "The fine-tuning smoke test failed with the default preparation. One more try with the peft helper preparation."
  Invoke-Stage -Name "s3b_smoke_sft_kbit" -PyArgs @("scripts/train_sft.py", "--seed", "0", "--train", "data/smoke_train.jsonl", "--limit", "8", "--max_steps", "4", "--ckpt_every", "2", "--probe_file", "data/train.jsonl", "--out", $smokeOutKbit, "--prep", "kbit_helper", "--backend", $Backend)
  if ($script:stageOk) {
    $prep = "kbit_helper"
    $smokeAdapter = $smokeOutKbit
  }
}
$smoke["sft"] = $script:stageOk
$smoke["prep"] = $prep
$smoke["adapter"] = $smokeAdapter
Save-Json -Object $smoke -Path $smokeFile

if ($smoke["sft"]) {
  Invoke-Stage -Name "s4_smoke_sft_infer" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_4b_2507", "--items", "data/smoke_test.jsonl", "--variants", "V2", "--adapter", $smokeAdapter, "--limit", "3", "--backend", $Backend)
  $smoke["sft_infer"] = $script:stageOk
} else {
  Write-Host ""
  Write-Host "=== SKIPPED s4_smoke_sft_infer : no smoke adapter"
}
Save-Json -Object $smoke -Path $smokeFile

# The reward check always uses the default preparation (lean). With the peft helper preparation it needs about 6.6 GB,
# which this GPU does not have; Windows would then use shared system memory and the 150 steps would crawl for hours.
# If lean does not work, this smoke test fails within two minutes and only the reward check is skipped.
Invoke-Stage -Name "s5_smoke_grpo" -PyArgs @("scripts/grpo_check.py", "--steps", "2", "--limit", "4", "--prep", "lean", "--backend", $Backend)
$smoke["grpo"] = $script:stageOk
Save-Json -Object $smoke -Path $smokeFile

Write-Host ""
Write-Host ("SMOKE RESULT: infer_8b=" + $smoke["infer_8b"] + " infer_4b=" + $smoke["infer_4b"] + " sft=" + $smoke["sft"] + " sft_infer=" + $smoke["sft_infer"] + " grpo=" + $smoke["grpo"] + " prep=" + $prep)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Full run. Cheap and certain stages first. A stage whose smoke test failed is skipped; the others still run.
# ---------------------------------------------------------------------------------------------------------------------
$full = [ordered]@{ started = (Get-Date -Format "yyyy-MM-dd HH:mm:ss"); backend = $Backend; prep = $prep }
$skipSmoke = "skipped_smoke_failed"
$skipAdapter = "skipped_no_adapter"
$gate8b = [bool]($smoke["infer_8b"])
$gate4b = [bool]($smoke["infer_4b"])
$gateSft = [bool]($smoke["sft"])
$gateSftInfer = [bool]($smoke["sft_infer"])
$gateGrpo = [bool]($smoke["grpo"])
if (-not $commitOk) {
  Write-Host "ERROR: P4_COMMIT.txt is missing or invalid: every stage of the full run is skipped."
  $skipSmoke = "skipped_no_commit_file"
  $skipAdapter = "skipped_no_commit_file"
  $gate8b = $false
  $gate4b = $false
  $gateSft = $false
  $gateSftInfer = $false
  $gateGrpo = $false
}
$out1 = $adp + "/sft_s1"
$out2 = $adp + "/sft_s2"
$out3 = $adp + "/sft_s3"

Invoke-FullStage -Key "closedbook_8b" -Gate $gate8b -SkipWord $skipSmoke -Name "f1_closedbook_8b" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_8b", "--items", "data/closedbook_items.jsonl", "--variants", "CB", "--backend", $Backend)

Invoke-FullStage -Key "base_8b_v1" -Gate $gate8b -SkipWord $skipSmoke -Name "f2_base_8b_v1" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_8b", "--items", "data/items_test.jsonl", "--variants", "V1", "--backend", $Backend)

Invoke-FullStage -Key "base_4b_v1v2" -Gate $gate4b -SkipWord $skipSmoke -Name "f3_base_4b_v1v2" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_4b_2507", "--items", "data/items_test.jsonl", "--variants", "V1,V2", "--backend", $Backend)

# A fine-tuning stage that failed is started once more: train_sft.py continues from its last checkpoint (every 50 steps).
Invoke-FullStage -Key "sft_s1" -Gate $gateSft -SkipWord $skipSmoke -Name "f4_sft_s1" -PyArgs @("scripts/train_sft.py", "--seed", "1", "--out", $out1, "--prep", $prep, "--backend", $Backend)
if ($full["sft_s1"] -eq "failed") {
  $full["sft_s1_first_try"] = "failed"
  Invoke-FullStage -Key "sft_s1" -Gate $gateSft -SkipWord $skipSmoke -Name "f4b_sft_s1_retry" -PyArgs @("scripts/train_sft.py", "--seed", "1", "--out", $out1, "--prep", $prep, "--backend", $Backend)
}
$gateEval = ($gateSftInfer -and ($full["sft_s1"] -eq "ok"))
$skipEval = $skipAdapter
if (-not $gateSftInfer) { $skipEval = $skipSmoke }
Invoke-FullStage -Key "eval_sft_s1" -Gate $gateEval -SkipWord $skipEval -Name "f5_eval_sft_s1" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_4b_2507", "--items", "data/items_test.jsonl", "--variants", "V2", "--adapter", $out1, "--backend", $Backend)

Invoke-FullStage -Key "sft_s2" -Gate $gateSft -SkipWord $skipSmoke -Name "f6_sft_s2" -PyArgs @("scripts/train_sft.py", "--seed", "2", "--out", $out2, "--prep", $prep, "--backend", $Backend)
if ($full["sft_s2"] -eq "failed") {
  $full["sft_s2_first_try"] = "failed"
  Invoke-FullStage -Key "sft_s2" -Gate $gateSft -SkipWord $skipSmoke -Name "f6b_sft_s2_retry" -PyArgs @("scripts/train_sft.py", "--seed", "2", "--out", $out2, "--prep", $prep, "--backend", $Backend)
}
$gateEval = ($gateSftInfer -and ($full["sft_s2"] -eq "ok"))
Invoke-FullStage -Key "eval_sft_s2" -Gate $gateEval -SkipWord $skipEval -Name "f7_eval_sft_s2" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_4b_2507", "--items", "data/items_test.jsonl", "--variants", "V2", "--adapter", $out2, "--backend", $Backend)

Invoke-FullStage -Key "sft_s3" -Gate $gateSft -SkipWord $skipSmoke -Name "f8_sft_s3" -PyArgs @("scripts/train_sft.py", "--seed", "3", "--out", $out3, "--prep", $prep, "--backend", $Backend)
if ($full["sft_s3"] -eq "failed") {
  $full["sft_s3_first_try"] = "failed"
  Invoke-FullStage -Key "sft_s3" -Gate $gateSft -SkipWord $skipSmoke -Name "f8b_sft_s3_retry" -PyArgs @("scripts/train_sft.py", "--seed", "3", "--out", $out3, "--prep", $prep, "--backend", $Backend)
}
$gateEval = ($gateSftInfer -and ($full["sft_s3"] -eq "ok"))
Invoke-FullStage -Key "eval_sft_s3" -Gate $gateEval -SkipWord $skipEval -Name "f9_eval_sft_s3" -PyArgs @("scripts/run_infer.py", "--model", "qwen3_4b_2507", "--items", "data/items_test.jsonl", "--variants", "V2", "--adapter", $out3, "--backend", $Backend)

Invoke-FullStage -Key "grpo" -Gate $gateGrpo -SkipWord $skipSmoke -Name "f10_grpo_check" -PyArgs @("scripts/grpo_check.py", "--prep", "lean", "--backend", $Backend)

$full["finished"] = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Save-Json -Object $full -Path $fullFile

# ---------------------------------------------------------------------------------------------------------------------
# 3. Zip of the results. pack_results.py prints the summary in Russian and the name of the zip to send back.
# ---------------------------------------------------------------------------------------------------------------------
if ($transcriptOn) {
  try { Stop-Transcript | Out-Null } catch { }
}
Invoke-Stage -Name "99_pack_results" -PyArgs @("scripts/pack_results.py", "--backend", $Backend)
if (-not $script:stageOk) {
  Write-Host "The zip was NOT written. Send the whole folder 'results' of this package instead (README_RU.md)."
}
