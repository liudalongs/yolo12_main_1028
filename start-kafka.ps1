# start-kafka.ps1
# Kafka KRaft Mode Startup Script for Windows (English version)

param(
    [string]$LogDir = "D:\kafka\data\kraft-logs",
    [string]$ConfigFile = "config\kraft\server.properties"
)

Write-Host "=== Kafka KRaft Mode Startup Script ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date)"
Write-Host "Working directory: $(Get-Location)"
Write-Host ""

function Cleanup-Logs {
    if (Test-Path $LogDir) {
        Write-Host "Cleaning existing log directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $LogDir
        Write-Host "✓ Log directory cleaned: $LogDir" -ForegroundColor Green
    } else {
        Write-Host "Log directory does not exist, skip cleanup." -ForegroundColor Gray
    }
}

function Format-Storage {
    Write-Host ""
    Write-Host "=== Step 1: Generate Cluster ID ===" -ForegroundColor Cyan
    $ClusterId = .\bin\windows\kafka-storage.bat random-uuid 2>&1 | Out-String
    $ClusterId = $ClusterId.Trim()
    Write-Host "Generated Cluster ID: $ClusterId"

    Write-Host ""
    Write-Host "=== Step 2: Format Storage Directory ===" -ForegroundColor Cyan
    & .\bin\windows\kafka-storage.bat format -t $ClusterId -c $ConfigFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Storage formatted successfully." -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Storage formatting failed." -ForegroundColor Red
        return $false
    }
}

function Start-KafkaServer {
    Write-Host ""
    Write-Host "=== Step 3: Starting Kafka Server ===" -ForegroundColor Cyan
    Write-Host "Config file: $ConfigFile"
    Write-Host "Starting Kafka... (Press Ctrl+C to stop)" -ForegroundColor Yellow
    Write-Host ""
    .\bin\windows\kafka-server-start.bat $ConfigFile
}

# Validate required files
if (-not (Test-Path $ConfigFile)) {
    Write-Host "✗ Config file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\bin\windows\kafka-storage.bat")) {
    Write-Host "✗ kafka-storage.bat not found. Run this script from Kafka root directory." -ForegroundColor Red
    exit 1
}

# Main logic
$success = Format-Storage
if (-not $success) {
    Write-Host ""
    Write-Host "⚠️ Formatting failed. Cleaning logs and retrying..." -ForegroundColor Yellow
    Cleanup-Logs
    Start-Sleep -Seconds 1
    $success = Format-Storage
    if (-not $success) {
        Write-Host "✗ Retry failed. Please check configuration." -ForegroundColor Red
        exit 1
    }
}

Start-KafkaServer