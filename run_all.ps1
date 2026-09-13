param(
    [switch]$TrainModel,
    [switch]$SkipInstall,
    [switch]$RefreshFromRaw
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    Write-Host ("`n==> {0}" -f $Description)
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw ("Step failed: {0} (exit code {1})" -f $Description, $LASTEXITCODE)
    }
}

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw ("Virtual environment creation failed (exit code {0})" -f $LASTEXITCODE)
    }
}

$python = (Resolve-Path '.venv\Scripts\python.exe').Path
if (-not $SkipInstall) {
    Invoke-Step "Install base dependencies" { & $python -m pip install -r requirements.txt }
}

$requiredCleanFiles = @(
    'data\clean\occupations.csv',
    'data\clean\skills.csv',
    'data\clean\occupation_skill.csv',
    'data\clean\resumes_zh.csv'
)
$missingCleanFiles = @($requiredCleanFiles | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($RefreshFromRaw -or $missingCleanFiles.Count -gt 0) {
    Invoke-Step "Prepare dataset from O*NET raw files" { & $python src\prepare_dataset.py }
} else {
    Write-Host "`n==> Reuse checked-in clean dataset (use -RefreshFromRaw to rebuild)"
}
Invoke-Step "Generate skill forecasts" { & $python src\forecast_skills.py }
Invoke-Step "Build features" { & $python src\build_features.py }
Invoke-Step "Evaluate recommendations" { & $python src\evaluate_recommendation.py }
Invoke-Step "Initialize SQLite database" { & $python src\init_database.py --db-path artifacts\topic17.sqlite3 --data-dir data\clean }
Invoke-Step "Build occupation-skill bipartite graph" { & $python src\bipartite_graph.py --data-dir data\clean --out-dir data\processed\bipartite --min-demand-weight 0.30 }
if ($TrainModel) {
    Invoke-Step "Train TemporalGAT" { & $python src\train_temporal_gat.py --data-dir data\clean --artifact-dir artifacts\models --epochs 30 --seed 42 }
}
Invoke-Step "Run stage verification" { & $python src\verify_stage1.py --db-path artifacts\topic17.sqlite3 }
Write-Host "CareerGraph Recommender data, database, feature, bipartite graph, and verification pipeline completed."
exit 0
