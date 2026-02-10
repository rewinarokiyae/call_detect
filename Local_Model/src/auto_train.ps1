$TargetCount = 150000 
$PrecomputedDir = "D:\Subject\honors\ai_project\call_detect\data\precomputed\train"
$TrainScript = "Local_Model\src\train_local.py"
$PythonPath = ".venv312\Scripts\python.exe"

Write-Host "Started Auto-Trainer Monitor..." -ForegroundColor Cyan
Write-Host "Waiting for feature extraction to reach $TargetCount files..."

while ($true) {
    if (Test-Path $PrecomputedDir) {
        $files = (Get-ChildItem -Path $PrecomputedDir -Filter "*.npy" | Measure-Object).Count
        Write-Host "Current Count: $files" -NoNewline -ForegroundColor Yellow
        Write-Host "`r" -NoNewline
        
        if ($files -ge $TargetCount) {
            Write-Host "`nTarget reached! ($files files)." -ForegroundColor Green
            Write-Host "Starting Training..." -ForegroundColor Cyan
            
            # Start Training
            & $PythonPath $TrainScript
            
            Write-Host "Training Process Finished." -ForegroundColor Green
            break
        }
    } else {
        Write-Host "Waiting for directory creation..."
    }
    
    Start-Sleep -Seconds 60
}
