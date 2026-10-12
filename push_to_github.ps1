# Pushes this project to https://github.com/Naveena0702/beijing-pm25-eda-lstm
# Usage: right-click the folder > Open in Terminal, then run:
#   powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1

$ErrorActionPreference = "Stop"
$User = "Naveena0702"
$Repo = "beijing-pm25-eda-lstm"
$Url  = "https://github.com/$User/$Repo.git"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Install it from https://git-scm.com/download/win and run this script again." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".git")) {
    Write-Host "The .git folder is missing. Extract the whole zip (including hidden files) and try again." -ForegroundColor Red
    exit 1
}

# Commits in this repository are already authored by Naveena B <bnaveena2007@gmail.com>
git config user.name  "Naveena B"
git config user.email "bnaveena2007@gmail.com"
git branch -M main

if (Get-Command gh -ErrorAction SilentlyContinue) {
    # GitHub CLI available: log in (browser opens once) and create the repository automatically
    gh auth status 2>$null
    if ($LASTEXITCODE -ne 0) { gh auth login --hostname github.com --git-protocol https --web }
    gh repo view "$User/$Repo" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        gh repo create "$User/$Repo" --public --description "EDA and LSTM forecasting of Beijing PM2.5 (UCI 501) - U21ADP05 Assignment II"
    }
} else {
    Write-Host ""
    Write-Host "1. Open https://github.com/new while logged in as $User" -ForegroundColor Yellow
    Write-Host "2. Repository name: $Repo   Visibility: Public   Do NOT add a README, .gitignore or licence" -ForegroundColor Yellow
    Write-Host "3. Click 'Create repository', then come back here." -ForegroundColor Yellow
    Read-Host "Press Enter once the empty repository exists"
}

if (git remote | Select-String -Quiet "^origin$") { git remote set-url origin $Url } else { git remote add origin $Url }
git push -u origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done: https://github.com/$User/$Repo" -ForegroundColor Green
} else {
    Write-Host "Push failed. If a sign-in window appeared, sign in as $User and run the script again." -ForegroundColor Red
}
