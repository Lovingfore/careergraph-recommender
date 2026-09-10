param(
    [switch]$TrainModel
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
}

$python = (Resolve-Path '.venv\Scripts\python.exe').Path
& $python -m pip install -r requirements.txt
& $python src\prepare_dataset.py
& $python src\forecast_skills.py
& $python src\build_features.py
& $python src\evaluate_recommendation.py
& $python src\init_database.py --db-path artifacts\topic17.sqlite3 --data-dir data\clean
& $python src\bipartite_graph.py --data-dir data\clean --out-dir data\processed\bipartite --min-demand-weight 0.30
if ($TrainModel) {
    & $python src\train_temporal_gat.py --data-dir data\clean --artifact-dir artifacts\models --epochs 30 --seed 42
}
& $python src\verify_stage1.py --db-path artifacts\topic17.sqlite3
Write-Host "CareerGraph Recommender data, database, feature, bipartite graph, and verification pipeline completed."
