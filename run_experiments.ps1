# -----------------------------------------------------------------------------
# Automation Script: Academic Multi-Run Experiment
# Logs 5 consecutive runs of main.py (Combo Mode).
# -----------------------------------------------------------------------------

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Starting RL Multi-Run (Ablation Statistics)" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$TotalRuns = 5

for ($i = 1; $i -le $TotalRuns; $i++) {
    Write-Host "`n>>> [RUN $i / $TotalRuns] Executing iteration..." -ForegroundColor Yellow
    
    # Execute main program
    python main.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> [ERROR] Run $i failed! Stopping script." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host ">>> [RUN $i / $TotalRuns] Iteration $i completed successfully!" -ForegroundColor Green
}

Write-Host "`n==============================================" -ForegroundColor Cyan
Write-Host "  Finished! 5-run pair experiments completed." -ForegroundColor Cyan
Write-Host "  Use: tensorboard --logdir logs_and_results" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
