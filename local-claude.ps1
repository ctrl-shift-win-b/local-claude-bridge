param(
    [switch]$DebugBridge,
    [switch]$NoPoke,
    [switch]$VisionInternal,
    [string]$VisionExternal,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$PassThruArgs
)

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

# double-dash flags from cmd.exe are not parsed as switches; strip them manually
if ($PassThruArgs -contains "--no-poke") {
    $NoPoke = $true
    $PassThruArgs = $PassThruArgs | Where-Object { $_ -ne "--no-poke" }
}
if ($PassThruArgs -contains "--debug-bridge") {
    $DebugBridge = $true
    $PassThruArgs = $PassThruArgs | Where-Object { $_ -ne "--debug-bridge" }
}

function Kill-Port([int]$port) {
    $lines = & netstat -ano
    foreach ($line in $lines) {
        if ($line -match "TCP\s+\S+:$port\s+\S+\s+LISTENING\s+(\d+)") {
            Stop-Process -Id ([int]$Matches[1]) -Force -ErrorAction SilentlyContinue
        }
    }
}

function Wait-Health([string]$url, [int]$intervalSec) {
    while ($true) {
        $code = & curl.exe -s -o NUL -w "%{http_code}" --max-time 2 $url 2>$null
        if ($code -eq "200") { return }
        Start-Sleep -Seconds $intervalSec
    }
}

# Kill any stale processes from a previous session
Kill-Port 1235
Kill-Port 1234

$serverProc = $null
$bridgeProc = $null

try {
    # Launch llama.cpp server in a new window
    $serverProc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c `"$dir\start_server.bat`"" `
        -WorkingDirectory $dir `
        -PassThru

    Write-Host "Waiting for llama.cpp server..."
    Wait-Health "http://localhost:1234/health" 2
    Write-Host "llama.cpp ready."

    # Launch bridge in a new window
    $bridgeArgs = "`"$dir\bridge.py`""
    if ($DebugBridge)   { $bridgeArgs += " --debug" }
    if ($NoPoke)        { $bridgeArgs += " --no-poke" }
    if ($VisionInternal){ $bridgeArgs += " --image-processing-internal" }
    if ($VisionExternal){ $bridgeArgs += " --image-processing-external `"$VisionExternal`"" }
    $bridgeProc = Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList $bridgeArgs `
        -WorkingDirectory $dir `
        -PassThru

    Write-Host "Waiting for bridge..."
    Wait-Health "http://localhost:1235/health" 1
    Write-Host "Bridge ready."

    $env:ANTHROPIC_BASE_URL             = "http://localhost:1235"
    $env:ANTHROPIC_AUTH_TOKEN           = "local"
    $env:ANTHROPIC_MODEL                = "local-model"
    $env:CLAUDE_CODE_ATTRIBUTION_HEADER = "0"
    $env:CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "65"

    if ($PassThruArgs) {
        # & claude --debug @PassThruArgs
        & claude @PassThruArgs
    } else {
        # & claude --debug
        & claude
    }

} finally {
    Write-Host "`nShutting down..."
    Kill-Port 1235
    Kill-Port 1234
    if ($bridgeProc -and -not $bridgeProc.HasExited) {
        Stop-Process -Id $bridgeProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($serverProc -and -not $serverProc.HasExited) {
        Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Done."
}
