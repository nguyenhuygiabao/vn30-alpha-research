<#
.SYNOPSIS
    Run the VN30 model and show where it currently stands.

.DESCRIPTION
    Four modes, cheapest first.

      status       Print the current headline numbers. Seconds. Runs nothing.
      costs        Regenerate the cost-dependent stages. Tens of minutes.
      full         Regenerate everything including model refits. Hours.
      tests        Run the test suite. About two minutes.

    "costs" is the one to use after a change to the cost model: it reruns
    baseline, risk model, optimizer, backtester, ablation, horizons and the
    report tables, while reusing cached model fits, which the cost model does
    not affect.

.EXAMPLE
    .\run_model.ps1
    .\run_model.ps1 -Mode costs
    .\run_model.ps1 -Mode full
#>
param(
    [ValidateSet('status', 'costs', 'full', 'tests')]
    [string]$Mode = 'status'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Write-Head($text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
}

if ($Mode -eq 'status') {
    Write-Head "VN30 MODEL STATUS"
    Write-Host "Repo: $root"
    Write-Host "Branch: $(git branch --show-current)"
    Write-Host ""

    $tranche = Join-Path $root 'reports\tables\tranche_portfolio_diagnostic.csv'

    if (Test-Path $tranche) {
        Write-Host "DEPLOYABILITY - active equity vs VN30, 1.0 = matches the index" -ForegroundColor Yellow
        Write-Host "(corrected cost model: commission and slippage both legs, sale tax)"
        Write-Host ""

        $rows = Import-Csv $tranche | Where-Object {
            $_.execution_mode -eq 'price_limit_aware_tranche'
        } | Sort-Object { [double]$_.active_final_equity_after_cost } -Descending

        $fmt = "{0,-20} {1,10} {2,10} {3,12}"
        Write-Host ($fmt -f 'optimizer', 'active', 'before-cost', 'drawdown %') -ForegroundColor DarkGray

        foreach ($r in $rows) {
            $active = [double]$r.active_final_equity_after_cost
            $colour = if ($active -ge 1.0) { 'Green' } else { 'Red' }
            Write-Host ($fmt -f `
                $r.optimization_mode,
                $active.ToString('0.0000'),
                ([double]$r.active_final_equity_before_cost).ToString('0.0000'),
                ([double]$r.active_max_drawdown_percent_after_cost).ToString('0.0')
            ) -ForegroundColor $colour
        }

        Write-Host ""
        Write-Host "Note: this is the 5-day rebalance. Bootstrap says it is" -ForegroundColor DarkGray
        Write-Host "decisively worse than the index, 95% CI [0.3656, 0.8519]." -ForegroundColor DarkGray
        Write-Host "A 20-day rebalance reaches ~1.03 but its interval spans 1.0." -ForegroundColor DarkGray
    }
    else {
        Write-Host "No deployability table yet. Run: .\run_model.ps1 -Mode costs" -ForegroundColor Yellow
    }

    Write-Head "STALE ARTIFACTS"
    foreach ($name in @('horizon_results.csv', 'ablation_results.csv')) {
        $p = Join-Path $root "reports\tables\$name"
        if (Test-Path $p) {
            $age = (Get-Item $p).LastWriteTime
            Write-Host ("{0,-28} last written {1}" -f $name, $age)
        }
    }
    Write-Host ""
    Write-Host "If those predate the cost-model fix, their after-cost columns" -ForegroundColor DarkGray
    Write-Host "were computed at 2.5x too little cost. Rank IC is unaffected." -ForegroundColor DarkGray
    Write-Host "Regenerate with: .\run_model.ps1 -Mode costs" -ForegroundColor DarkGray

    Write-Head "IDEAS LOG"
    $intraday = "C:\Users\HUY BAO\quant-projects\vn30-alpha-research-intraday"
    $candidates = @(
        (Join-Path $intraday 'IDEAS.txt'),
        (Join-Path $intraday '.claude\worktrees\phase0-intraday-survey\IDEAS.txt')
    )
    $ideas = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($ideas) {
        Select-String -Path $ideas -Pattern '^IDEA \d+' | ForEach-Object {
            $line = $_.Line
            $colour = 'Gray'
            if ($line -match 'TESTED') { $colour = 'Yellow' }
            if ($line -match 'RUNNING') { $colour = 'Cyan' }
            if ($line -match 'EVIDENCE') { $colour = 'Green' }
            if ($line -match 'DEAD') { $colour = 'DarkGray' }
            Write-Host ("  " + $line) -ForegroundColor $colour
        }
        Write-Host ""
        Write-Host "  Full log: $ideas" -ForegroundColor DarkGray
    }
    else {
        Write-Host "  (no ideas log found)" -ForegroundColor DarkGray
    }

    Write-Host ""
    exit 0
}

if ($Mode -eq 'tests') {
    Write-Head "TEST SUITE"
    py -m pytest -q
    exit $LASTEXITCODE
}

if ($Mode -eq 'costs') {
    Write-Head "REGENERATING COST-DEPENDENT STAGES"
    Write-Host "Reuses cached model fits. The cost model does not affect them."
    Write-Host "Expect tens of minutes; ablation and horizons are the slow parts."
    Write-Host ""
    py scripts\run_full_pipeline.py --reuse-existing-ml --start-at baseline_portfolios
    exit $LASTEXITCODE
}

if ($Mode -eq 'full') {
    Write-Head "FULL PIPELINE"
    Write-Host "Refits every model across ~2,418 walk-forward windows."
    Write-Host "This takes hours. Use -Mode costs unless you changed the models."
    Write-Host ""
    py scripts\run_full_pipeline.py
    exit $LASTEXITCODE
}
