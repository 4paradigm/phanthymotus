param(
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$installDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $installDir ".venv\Scripts\python.exe"
$runScript = Join-Path $installDir "run_bridge.py"
$configPath = Join-Path $installDir "config.json"
$watchdogLog = Join-Path $installDir "data\watchdog.log"
$healthUrl = "http://127.0.0.1:8765/health"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $watchdogLog) | Out-Null

function Write-WatchdogLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $watchdogLog -Value $line -Encoding UTF8
}

function Test-BridgeHealth {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return $response.state -eq "running"
    }
    catch {
        return $false
    }
}

function Start-BridgeIfNeeded {
    if (Test-BridgeHealth) {
        return $true
    }

    if (-not (Test-Path -LiteralPath $pythonExe) -or
        -not (Test-Path -LiteralPath $runScript) -or
        -not (Test-Path -LiteralPath $configPath)) {
        Write-WatchdogLog "启动失败：程序、虚拟环境或配置文件缺失。"
        return $false
    }

    $listeners = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        try {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
            if ($process.CommandLine -like "*run_bridge.py*") {
                Stop-Process -Id $listener.OwningProcess -Force
                Write-WatchdogLog "停止了无响应的桥接进程 PID=$($listener.OwningProcess)。"
            }
            else {
                Write-WatchdogLog "端口 8765 被其他程序占用，未自动终止 PID=$($listener.OwningProcess)。"
                return $false
            }
        }
        catch {
            Write-WatchdogLog "检查端口进程失败：$($_.Exception.Message)"
            return $false
        }
    }

    Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @($runScript, "--config", $configPath) `
        -WorkingDirectory $installDir `
        -WindowStyle Hidden

    for ($attempt = 0; $attempt -lt 15; $attempt++) {
        Start-Sleep -Seconds 1
        if (Test-BridgeHealth) {
            Write-WatchdogLog "桥接服务已自动启动。"
            return $true
        }
    }

    Write-WatchdogLog "桥接服务启动后 15 秒内未通过健康检查。"
    return $false
}

do {
    try {
        Start-BridgeIfNeeded | Out-Null
    }
    catch {
        Write-WatchdogLog "监护检查异常：$($_.Exception.Message)"
    }
    if (-not $Once) {
        Start-Sleep -Seconds 5
    }
} while (-not $Once)
