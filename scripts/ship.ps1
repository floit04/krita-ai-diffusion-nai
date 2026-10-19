<#
.SYNOPSIS
  One command: verify, commit, push, and install the NAI fork into Krita.

.DESCRIPTION
  The steps this replaces were run by hand every round and in this order,
  because each one only means something if the previous one passed:

    1. pytest        - the NAI suite plus the novelai.net golden samples
    2. ruff          - format then lint
    3. git commit    - everything not ignored
    4. git push      - to the current branch's remote
    5. install       - copy ai_diffusion into Krita's pykrita, with a backup

  Nothing is pushed if the tests or the linter fail. The install runs after the
  push rather than before, so what Krita ends up running is what GitHub has.

.PARAMETER Message
  Commit message. If the working tree is clean this is ignored and the script
  goes straight to pushing whatever is unpushed.

.PARAMETER SkipTests
  Push without running pytest or ruff. For when the tests were green a minute
  ago and you are in a hurry - it prints a warning so the omission is on record.

.PARAMETER NoInstall
  Push only; leave Krita's copy alone.

.PARAMETER NoPush
  Verify, commit and install, but do not push. For working offline.

.EXAMPLE
  .\scripts\ship.ps1 "NAI: fix the thing"

.EXAMPLE
  .\scripts\ship.ps1 -SkipTests -NoInstall "wip"
#>
param(
    [Parameter(Position = 0)]
    [string]$Message,
    [switch]$SkipTests,
    [switch]$NoInstall,
    [switch]$NoPush
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Fail($text) { Write-Host "FAILED: $text" -ForegroundColor Red; exit 1 }

# Windows PowerShell 5.1 turns ANY native stderr output into a terminating error
# while $ErrorActionPreference is 'Stop', and git sends its progress and its
# line-ending warnings there. Those are not failures, so native commands run
# through Run, which captures both streams and judges them by $LASTEXITCODE.
function Run($text, [scriptblock]$command) {
    # 'Continue' inside: with 'Stop', PS 5.1 aborts on the ErrorRecord that the
    # 2>&1 merge produces, before Out-String can collect it.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $command 2>&1 | Out-String
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host $output
        Fail $text
    }
    Write-Host $output.TrimEnd()
}

# PYTHONUTF8 is not optional here: the tests carry Chinese comments and the
# golden samples carry Japanese tags, and the default Windows codepage mangles
# both into a UnicodeDecodeError that looks like a test failure.
$env:PYTHONUTF8 = '1'

# --- 1. Tests -------------------------------------------------------------
if ($SkipTests) {
    Write-Host "WARNING: tests and linter skipped (-SkipTests)" -ForegroundColor Yellow
}
else {
    Step 'pytest'
    # test_client / test_workflow / test_image_transfer need a live ComfyUI
    # server, which this fork does not use. Everything else must pass.
    python -m pytest tests/ -q --ignore=tests/test_client.py `
        --ignore=tests/test_workflow.py --ignore=tests/test_image_transfer.py
    if ($LASTEXITCODE -ne 0) { Fail 'pytest' }

    Step 'ruff format'
    python -m ruff format .
    if ($LASTEXITCODE -ne 0) { Fail 'ruff format' }

    Step 'ruff check'
    # Three pre-existing errors live in upstream files this fork does not touch
    # (resources.py, updates.py, ui/settings.py). Reporting them every run would
    # train you to ignore the linter, so only NEW ones stop the ship.
    $lint = python -m ruff check . 2>&1 | Out-String
    $known = 'resources.py|updates.py|ui[\\/]settings.py'
    $new = $lint -split "`n" | Where-Object { $_ -match '^\S+\.py:\d+:\d+:' -and $_ -notmatch $known }
    if ($new) {
        Write-Host $($new -join "`n") -ForegroundColor Red
        Fail 'ruff check found new errors'
    }
    Write-Host 'no new lint errors' -ForegroundColor Green
}

# --- 2. Commit ------------------------------------------------------------
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$dirty = git status --porcelain

if ($dirty) {
    Step "git commit  (branch: $branch)"
    git status --short
    if (-not $Message) {
        Fail 'working tree is dirty but no commit message was given'
    }
    Run 'git add' { git add -A }
    Run 'git commit' { git commit -m $Message }
}
else {
    Write-Host "`nworking tree clean, nothing to commit" -ForegroundColor DarkGray
}

# --- 3. Push --------------------------------------------------------------
if ($NoPush) {
    Write-Host "`npush skipped (-NoPush)" -ForegroundColor Yellow
}
else {
    Step "git push origin $branch"
    Run 'git push' { git push origin $branch }
}

# --- 4. Install into Krita ------------------------------------------------
if ($NoInstall) {
    Write-Host "`ninstall skipped (-NoInstall)" -ForegroundColor Yellow
}
else {
    $pykrita = Join-Path $env:APPDATA 'krita\pykrita'
    $target = Join-Path $pykrita 'ai_diffusion'
    if (-not (Test-Path $pykrita)) {
        Write-Host "`nKrita pykrita folder not found at $pykrita - install skipped" -ForegroundColor Yellow
    }
    else {
        Step "install -> $target"
        if (Test-Path $target) {
            # Timestamped rather than a single rolling backup: two bad ships in a
            # row would otherwise overwrite the last known-good copy with a broken
            # one, which is exactly when you need it.
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $backup = Join-Path $pykrita "ai_diffusion.bak-$stamp"
            Move-Item $target $backup
            Write-Host "backup: $backup" -ForegroundColor DarkGray
        }
        # Copy from a clean export so local __pycache__ and any editor scratch
        # files never reach Krita - stale .pyc next to changed .py is a whole
        # afternoon of debugging a fix that is already installed.
        Run 'git archive / tar' { git archive HEAD ai_diffusion | tar -x -C $pykrita }

        # A syntax error here means Krita silently fails to load the plugin and
        # shows no docker at all, so catch it now rather than in the UI.
        Run 'py_compile on the installed copy' { python -m compileall -q $target }
        Write-Host 'installed and compiles' -ForegroundColor Green
    }
}

Step 'done'
git log --oneline -1Write-Host 'Restart Krita for the installed copy to take effect.' -ForegroundColor Yellow
