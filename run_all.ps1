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

function Find-SystemPython {
    $candidates = @(
        @{ Name = "py"; Args = @("-3.12") },
        @{ Name = "py"; Args = @("-3") },
        @{ Name = "python"; Args = @() },
        @{ Name = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Name -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            continue
        }
        & $command.Source @($candidate.Args) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }
    throw "No usable Python 3.10+ interpreter was found."
}

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $systemPython = Find-SystemPython
    & $systemPython.Name @($systemPython.Args) -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw ("Virtual environment creation failed (exit code {0})" -f $LASTEXITCODE)
    }
}

$python = (Resolve-Path '.venv\Scripts\python.exe').Path
& $python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv Python must be version 3.10 or newer."
}
if (-not $SkipInstall) {
    Invoke-Step "Install base dependencies" { & $python -m pip install -r requirements.txt }
    if ($TrainModel) {
        Invoke-Step "Install optional model dependencies" { & $python -m pip install -r requirements-optional-models.txt }
    }
}

$requiredCleanFiles = @(
    'data\clean\occupations.csv',
    'data\clean\skills.csv',
    'data\clean\occupation_skill.csv',
    'data\clean\resumes_zh.csv',
    'data\clean\job_transitions.csv',
    'data\clean\user_profiles.csv',
    'data\clean\user_skill_events.csv'
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
