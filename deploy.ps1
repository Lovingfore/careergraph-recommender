[CmdletBinding()]
param(
    [switch]$TrainModel,
    [switch]$StartWeb,
    [switch]$SkipInstall,
    [switch]$RefreshFromRaw,
    [Alias("Host")]
    [string]$ListenAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Set-Location -LiteralPath $projectRoot

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
    throw "No usable Python 3 was found. Install Python 3.10+ and add py/python to PATH."
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    Write-Host ("`n==> {0}" -f $Description) -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw ("Step failed: {0} (exit code {1})" -f $Description, $LASTEXITCODE)
    }
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($SkipInstall) {
        throw "The .venv folder is missing; remove -SkipInstall to create it and install dependencies."
    }
    $systemPython = Find-SystemPython
    Write-Host "Creating virtual environment: .venv" -ForegroundColor Cyan
    & $systemPython.Name @($systemPython.Args) -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw ("Virtual environment creation failed (exit code {0})" -f $LASTEXITCODE)
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The virtual environment was created but .venv\Scripts\python.exe was not found."
}

$python = (Resolve-Path -LiteralPath $venvPython).Path
$versionCheck = & $python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv Python must be version 3.10 or newer. Remove .venv and rerun without -SkipInstall."
}
if (-not $SkipInstall) {
    Invoke-PythonStep "Install base dependencies" { & $python -m pip install --disable-pip-version-check -r requirements.txt }
    if ($TrainModel) {
        Invoke-PythonStep "Install optional model dependencies" { & $python -m pip install --disable-pip-version-check -r requirements-optional-models.txt }
    }
}

$pipeline = Join-Path $projectRoot "run_all.ps1"
$pipelineArgs = @()
if ($TrainModel) {
    $pipelineArgs += "-TrainModel"
}
if ($SkipInstall) {
    $pipelineArgs += "-SkipInstall"
}
if ($RefreshFromRaw) {
    $pipelineArgs += "-RefreshFromRaw"
}
Invoke-PythonStep "Run data, feature, database, graph, and verification pipeline" {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $pipeline @pipelineArgs
}

Write-Host "`nDeployment preparation complete." -ForegroundColor Green
if ($StartWeb) {
    if ($ListenAddress -ne "127.0.0.1" -and $ListenAddress -ne "localhost") {
        $env:CAREERGRAPH_ALLOW_NETWORK = "1"
        Write-Host "Network binding enabled for this process (Django ALLOWED_HOSTS=*)" -ForegroundColor Yellow
    }
    Write-Host "Starting Web: http://$ListenAddress`:$Port/" -ForegroundColor Green
    & $python web\manage.py runserver "$ListenAddress`:$Port"
    if ($LASTEXITCODE -ne 0) {
        throw ("Web server exited (exit code {0})" -f $LASTEXITCODE)
    }
} else {
    if ($ListenAddress -ne "127.0.0.1" -and $ListenAddress -ne "localhost") {
        Write-Host ("Start Web: `$env:CAREERGRAPH_ALLOW_NETWORK='1'; & `"{0}`" web\manage.py runserver {1}:{2}" -f $python, $ListenAddress, $Port) -ForegroundColor Yellow
    } else {
        Write-Host ("Start Web: & `"{0}`" web\manage.py runserver {1}:{2}" -f $python, $ListenAddress, $Port) -ForegroundColor Yellow
    }
    Write-Host "To start automatically, add: -StartWeb" -ForegroundColor Yellow
}
