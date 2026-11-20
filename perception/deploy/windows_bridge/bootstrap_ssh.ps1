$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator SSH required' }

$upload = 'D:\ParadigmVisitorAutomation\upload'
$corePath = Join-Path $upload 'core.json'
$manifestPath = Join-Path $upload 'manifest.json'
if (!(Test-Path $corePath) -or !(Test-Path $manifestPath)) { throw 'Deployment files missing' }
$core = Get-Content -LiteralPath $corePath -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$root = 'D:\ParadigmVisitorAutomation'
$install = Join-Path $root 'app'
$assets = Join-Path $root 'offline'
$log = Join-Path $root 'install.log'
New-Item -ItemType Directory -Force -Path $install,$assets | Out-Null

$mutex = New-Object Threading.Mutex($false, 'Global\ParadigmVisitorAutomationInstall')
if (!$mutex.WaitOne(0)) { throw 'Another installation is running' }
try {
    foreach ($fileInfo in $manifest.files) {
        if ($fileInfo.name -notmatch '^[A-Za-z0-9_.-]+\.(exe|whl)$') { throw 'Invalid asset name' }
        $assetPath = Join-Path $assets $fileInfo.name
        $stream = [IO.FileStream]::new($assetPath, [IO.FileMode]::Create)
        try {
            for ($i = 0; $i -lt $fileInfo.chunk_count; $i++) {
                $chunk = Join-Path $upload ("{0}.{1:D4}.chunk" -f $fileInfo.name, $i)
                if (!(Test-Path $chunk)) { throw "Missing chunk: $chunk" }
                $bytes = [IO.File]::ReadAllBytes($chunk)
                $stream.Write($bytes, 0, $bytes.Length)
            }
        } finally { $stream.Dispose() }
        if ((Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash -ne $fileInfo.sha256) {
            throw 'Offline asset hash mismatch'
        }
    }

    $python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (!(Test-Path $python)) {
        $python = Join-Path $root 'Python312\python.exe'
        if (!(Test-Path $python)) {
            $installer = Join-Path $assets 'python-3.12.10-amd64.exe'
            $signature = Get-AuthenticodeSignature -LiteralPath $installer
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                throw 'Python installer signature invalid'
            }
            $process = Start-Process -FilePath $installer -ArgumentList @(
                '/quiet','InstallAllUsers=0',('TargetDir="' + (Join-Path $root 'Python312') + '"'),
                'Include_test=0','Include_launcher=0','Include_symbols=0','Include_debug=0','PrependPath=0'
            ) -Wait -PassThru -WindowStyle Hidden
            if ($process.ExitCode -notin @(0,3010)) { throw 'Python installation failed' }
        }
    }
    $venv = Join-Path $install '.venv'
    if (!(Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
        & $python -m venv $venv
        if ($LASTEXITCODE) { throw 'venv creation failed' }
    }
    $python = Join-Path $venv 'Scripts\python.exe'
    & $python -m pip --isolated install --no-index --no-cache-dir --disable-pip-version-check `
        --find-links $assets 'playwright==1.51.0' 'greenlet==3.1.1' 'pyee==12.1.1' 'typing_extensions==4.12.2' >> $log 2>&1
    if ($LASTEXITCODE) { throw 'Offline dependency installation failed' }

    $allowed = @('bridge_core.py','edge_backend.py','ssh_worker.py','login_edge.py','config.example.json','README.txt')
    foreach ($name in $allowed) {
        if (!$core.core_files.$name) { throw "Incomplete package: $name" }
        [IO.File]::WriteAllBytes((Join-Path $install $name), [Convert]::FromBase64String($core.core_files.$name))
    }
    & $python -m py_compile (Join-Path $install 'bridge_core.py') (Join-Path $install 'edge_backend.py') (Join-Path $install 'ssh_worker.py')
    if ($LASTEXITCODE) { throw 'Automation syntax validation failed' }

    $cfgPath = Join-Path $install 'config.json'
    if (!(Test-Path $cfgPath)) {
        $cfg = Get-Content (Join-Path $install 'config.example.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $cfg.PSObject.Properties.Remove('bind_host')
        $cfg.PSObject.Properties.Remove('port')
        $cfg.PSObject.Properties.Remove('shared_token')
        $cfg.database_path = 'data/automation.db'
        $cfg.edge.printer_name = ''
        $cfg | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $cfgPath -Encoding UTF8
    }

    # Remove the obsolete network bridge completely.
    Stop-ScheduledTask -TaskName 'Paradigm Visitor Bridge Watchdog' -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName 'Paradigm Visitor Bridge Watchdog' -Confirm:$false -ErrorAction SilentlyContinue
    Stop-ScheduledTask -TaskName 'Paradigm Visitor Bridge' -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName 'Paradigm Visitor Bridge' -Confirm:$false -ErrorAction SilentlyContinue
    Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
        if ($process.CommandLine -like '*ParadigmVisitorBridge*') { Stop-Process -Id $_.OwningProcess -Force }
    }
    Get-NetFirewallRule -Name 'ParadigmBridge-*' -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue

    $taskName = 'Paradigm Visitor Automation'
    $action = New-ScheduledTaskAction -Execute $python `
        -Argument ('"' + (Join-Path $install 'ssh_worker.py') + '" --config "' + $cfgPath + '"') `
        -WorkingDirectory $install
    $taskPrincipal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $taskPrincipal -Settings $settings -Force | Out-Null

    $loginTask = 'Paradigm Visitor Login'
    $loginAction = New-ScheduledTaskAction -Execute $python `
        -Argument ('"' + (Join-Path $install 'login_edge.py') + '" --automatic --config "' + $cfgPath + '"') `
        -WorkingDirectory $install
    Register-ScheduledTask -TaskName $loginTask -Action $loginAction -Principal $taskPrincipal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $loginTask

    New-Item -ItemType Directory -Force -Path (Join-Path $install 'data\jobs') | Out-Null
    Remove-Item -Recurse -Force $upload -ErrorAction SilentlyContinue
    Write-Output 'SSH_AUTOMATION_OK'
} catch {
    Add-Content -LiteralPath $log -Value $_.Exception.Message
    Write-Error 'SSH automation installation failed; see install.log'
    exit 1
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
