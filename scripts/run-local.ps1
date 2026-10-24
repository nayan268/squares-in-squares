param(
    [ValidateSet('quick','overnight')][string]$Preset = 'quick',
    [ValidateRange(1,4)][int]$Workers = 2,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunName = 'local-all-001',
    [switch]$Resume,
    [switch]$Wait
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repoRoot 'src/squares/local_run.py'))) {
    throw 'Project source was not found.'
}
$runDirectory = Join-Path $repoRoot ("runs/" + $RunName)
New-Item -ItemType Directory -Force $runDirectory | Out-Null
$settingsFile = Join-Path $runDirectory 'settings.json'
if ((Test-Path $settingsFile) -and -not $Resume) {
    throw 'Run already exists. Use -Resume or choose a new RunName.'
}
$lockFile = Join-Path $runDirectory 'active.lock'
if (Test-Path $lockFile) {
    throw 'A run lock exists. Stop or wait for that run; clear a stale lock only after verifying its process has ended.'
}
$stopFile = Join-Path $runDirectory 'STOP'
if ($Resume -and (Test-Path $stopFile)) { Remove-Item -LiteralPath $stopFile }
$env:PYTHONPATH = Join-Path $repoRoot 'src'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $runDirectory ("console-" + $stamp + ".log")
$errorPath = Join-Path $runDirectory ("errors-" + $stamp + ".log")
$arguments = @('-u','-m','squares','local-run','--preset',$Preset,
               '--workers',"$Workers",'--output',("runs/" + $RunName))
if ($Resume) { $arguments += '--resume' }
$launchOptions = @{FilePath=$pythonPath; ArgumentList=$arguments;
    WorkingDirectory=$repoRoot; WindowStyle='Hidden'; PassThru=$true;
    RedirectStandardOutput=$logPath; RedirectStandardError=$errorPath}
$childProcess = Start-Process @launchOptions
try { $childProcess.PriorityClass = 'BelowNormal' } catch { }
@{pid=$childProcess.Id; preset=$Preset; workers=$Workers; log=$logPath;
  errors=$errorPath; started=(Get-Date -Format o)} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDirectory 'process.json')
Write-Output ("Started local Python process " + $childProcess.Id)
Write-Output ("Progress: " + (Join-Path $runDirectory 'progress.json'))
Write-Output ("Log: " + $logPath)
Write-Output ("Final CSV: " + (Join-Path $runDirectory 'submission.csv'))

if ($Wait) { Wait-Process -Id $childProcess.Id }
