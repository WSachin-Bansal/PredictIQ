# Runs sensor / audio / fusion for several seeds on identical splits.
# Example:  powershell -File scripts/run_experiments.ps1 -Task severity -Split file
param(
  [string]$Task = "fault",       # fault | severity
  [string]$Split = "file",       # file | severity   (severity split needs -Task fault)
  [int[]]$Seeds = @(1, 2, 3),
  [int]$Epochs = 20,
  [int]$Workers = 2              # use 0 if DataLoader errors appear on Windows
)

foreach ($seed in $Seeds) {
  foreach ($mode in "sensor", "audio", "fusion") {
    Write-Host "=== mode=$mode task=$Task split=$Split seed=$seed ==="
    python -m predictiq.training.train_mafaulda --mode $mode --task $Task --split $Split `
      --seed $seed --epochs $Epochs --num-workers $Workers
    if ($LASTEXITCODE -ne 0) { Write-Error "run failed: $mode seed $seed"; exit 1 }
  }
}
python scripts/summarize_results.py
