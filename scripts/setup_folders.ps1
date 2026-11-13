# Creates the standard PredictIQ folder layout. Safe to re-run; never deletes anything.
# Run from the project root:   powershell -ExecutionPolicy Bypass -File scripts/setup_folders.ps1

$dirs = @(
  ".github/ISSUE_TEMPLATE",
  "configs",
  "data/raw/mafaulda", "data/raw/cmapss", "data/raw/mvtec", "data/processed",
  "checkpoints", "results",
  "docs", "notebooks", "reports/figures", "scripts",
  "dashboard", "tests",
  "src/predictiq/data", "src/predictiq/models", "src/predictiq/training",
  "src/predictiq/explain", "src/predictiq/api"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

# .gitkeep makes git track otherwise-empty folders, so everyone clones the same layout
foreach ($k in "data/raw", "data/processed", "checkpoints", "results", "notebooks", "reports/figures") {
  $f = Join-Path $k ".gitkeep"
  if (-not (Test-Path $f)) { New-Item -ItemType File -Path $f | Out-Null }
}

# Python package markers
$pkgs = @("src/predictiq", "src/predictiq/data", "src/predictiq/models",
          "src/predictiq/training", "src/predictiq/explain", "src/predictiq/api")
foreach ($p in $pkgs) {
  $f = Join-Path $p "__init__.py"
  if (-not (Test-Path $f)) { New-Item -ItemType File -Path $f | Out-Null }
}

Write-Host "Folder layout ready."
Get-ChildItem -Directory | Select-Object -ExpandProperty Name
