$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) {
        throw "未找到 Python。请安装 Python 3.10，并重新运行此脚本。"
    }
}

$versionText = & $python --version 2>&1
if ($versionText -notmatch "Python 3\.(9|10)\.") {
    throw "检测到 $versionText。当前固定依赖需要 Python 3.9 或 3.10，请使用兼容版本后重试。"
}

$streamlitCheck = & $python -c "import streamlit" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "正在安装 demo/requirements.txt ..." -ForegroundColor Yellow
    & $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
}

Set-Location $repoRoot
& $python -m streamlit run (Join-Path $PSScriptRoot "app.py") --server.address 127.0.0.1 --server.port 8522
