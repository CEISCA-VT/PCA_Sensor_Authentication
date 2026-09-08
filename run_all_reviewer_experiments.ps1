[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [string]$SourceDirectory = "",
    [string]$DatasetDirectory = "",
    [string]$ResultsDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 can evaluate parameter defaults before $PSScriptRoot
# is populated. Resolve the script directory after param(...) instead.
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = (Get-Location).Path
}
if ([string]::IsNullOrWhiteSpace($SourceDirectory)) {
    $SourceDirectory = $scriptRoot
}
if ([string]::IsNullOrWhiteSpace($DatasetDirectory)) {
    $DatasetDirectory = Join-Path $scriptRoot "01_master_dataset"
}
if ([string]::IsNullOrWhiteSpace($ResultsDirectory)) {
    $ResultsDirectory = Join-Path $scriptRoot "results"
}

function Resolve-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-LatestScript {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Stem
    )

    $escapedStem = [regex]::Escape($Stem)
    $pattern = "^$escapedStem(?:\s*\((\d+)\))?$"
    $candidates = @(
        Get-ChildItem -LiteralPath $Directory -File -Filter "*.py" |
            ForEach-Object {
                $match = [regex]::Match($_.BaseName, $pattern)
                if ($match.Success) {
                    $version = 0
                    if ($match.Groups[1].Success) {
                        $version = [int]$match.Groups[1].Value
                    }
                    [pscustomobject]@{
                        File = $_
                        Version = $version
                    }
                }
            } |
            Sort-Object -Property `
                @{ Expression = { $_.Version }; Descending = $true },
                @{ Expression = { $_.File.LastWriteTimeUtc }; Descending = $true }
    )

    if ($candidates.Count -eq 0) {
        throw "Could not find $Stem.py or $Stem(n).py in $Directory"
    }
    return $candidates[0].File
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $true)][string]$LogDirectory,
        [Parameter(Mandatory = $true)][int]$StepNumber,
        [Parameter(Mandatory = $true)][int]$StepCount
    )

    $label = [IO.Path]::GetFileNameWithoutExtension($ScriptName)
    $logPath = Join-Path $LogDirectory ("{0:D2}_{1}.log" -f $StepNumber, $label)
    Write-Host ""
    Write-Host ("[{0}/{1}] Running {2}" -f $StepNumber, $StepCount, $ScriptName) -ForegroundColor Cyan

    Push-Location $WorkingDirectory
    $oldErrorPreference = $ErrorActionPreference
    try {
        # Python warnings are captured in the log but do not stop PowerShell.
        $ErrorActionPreference = "Continue"
        & $PythonExe -u $ScriptName 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorPreference
        Pop-Location
    }

    if ($exitCode -ne 0) {
        throw "$ScriptName failed with exit code $exitCode. See $logPath"
    }
}

function Assert-FileExists {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required result was not produced: $Path"
    }
}

function Copy-CompactResult {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$DestinationDirectory,
        [Parameter(Mandatory = $true)][string]$DestinationName
    )
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        Copy-Item -LiteralPath $Source -Destination (Join-Path $DestinationDirectory $DestinationName) -Force
    }
}

$sourceRoot = Resolve-AbsolutePath $SourceDirectory
$datasetRoot = Resolve-AbsolutePath $DatasetDirectory
$pythonExe = (Get-Command $PythonCommand -ErrorAction Stop).Source

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsRoot = [IO.Path]::GetFullPath($ResultsDirectory)
$runDirectory = Join-Path $resultsRoot "run_$timestamp"
$logDirectory = Join-Path $runDirectory "logs"
$codeDirectory = Join-Path $runDirectory "code_used"
$consolidatedDirectory = Join-Path $runDirectory "00_consolidated_results"
$tableDirectory = Join-Path $consolidatedDirectory "tables"
$figureDirectory = Join-Path $consolidatedDirectory "figures"
$consolidatedLogDirectory = Join-Path $consolidatedDirectory "logs"

@(
    $runDirectory,
    $logDirectory,
    $codeDirectory,
    $consolidatedDirectory,
    $tableDirectory,
    $figureDirectory,
    $consolidatedLogDirectory
) | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force | Out-Null }

$scriptSpecifications = [ordered]@{
    "final_pca.py" = "final_pca"
    "two_stage.py" = "two_stage"
    "ds_pqm.py" = "ds_pqm"
    "kpca.py" = "kpca"
    "supervised.py" = "supervised"
    "unsup.py" = "unsup"
    "consolidate_nonlinear_comparison.py" = "consolidate_nonlinear_comparison"
}

$selectedScripts = [ordered]@{}
foreach ($canonicalName in $scriptSpecifications.Keys) {
    $sourceFile = Resolve-LatestScript -Directory $sourceRoot -Stem $scriptSpecifications[$canonicalName]
    $destination = Join-Path $runDirectory $canonicalName
    Copy-Item -LiteralPath $sourceFile.FullName -Destination $destination -Force
    Copy-Item -LiteralPath $sourceFile.FullName -Destination (Join-Path $codeDirectory $canonicalName) -Force
    $selectedScripts[$canonicalName] = [ordered]@{
        SourceFile = $sourceFile.FullName
        SHA256 = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
    }
}

# The analysis scripts require ./01_master_dataset relative to their working directory.
$datasetLink = Join-Path $runDirectory "01_master_dataset"
try {
    New-Item -ItemType Junction -Path $datasetLink -Target $datasetRoot | Out-Null
}
catch {
    throw "Could not create the dataset junction at $datasetLink. $($_.Exception.Message)"
}

$env:MPLBACKEND = "Agg"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Checking Python dependencies..." -ForegroundColor Cyan
& $pythonExe -c "import matplotlib, numpy, pandas, sklearn, torch; print('Python dependencies: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Missing Python dependencies. Install numpy, pandas, matplotlib, scikit-learn, and torch before running."
}

$executionOrder = @(
    "final_pca.py",
    "two_stage.py",
    "ds_pqm.py",
    "kpca.py",
    "supervised.py",
    "unsup.py",
    "consolidate_nonlinear_comparison.py"
)

$startedAt = Get-Date
for ($index = 0; $index -lt $executionOrder.Count; $index++) {
    Invoke-PythonStep `
        -PythonExe $pythonExe `
        -WorkingDirectory $runDirectory `
        -ScriptName $executionOrder[$index] `
        -LogDirectory $logDirectory `
        -StepNumber ($index + 1) `
        -StepCount $executionOrder.Count
}
$completedAt = Get-Date

# Required output checks.
$pcaSummaryPath = Join-Path $runDirectory "01_256reports_revised\comparison_summary.csv"
$twoStagePath = Join-Path $runDirectory "01_256reports_optimization_revised\optimization_results_128bit.csv"
$narrowVariancePath = Join-Path $runDirectory "01_256reports_optimization_revised\full_rank_pca_variance_10k-100k_128bit.csv"
$wideVariancePath = Join-Path $runDirectory "01_256reports_optimization_revised\full_rank_pca_variance_10k-1000k_128bit.csv"
$scalarSummaryPath = Join-Path $runDirectory "01_scalar_pqm_reports_revised\comparison_summary.csv"
$kpcaSummaryPath = Join-Path $runDirectory "01_kpca_reports_revised\comparison_summary.csv"
$supervisedSummaryPath = Join-Path $runDirectory "01_supervised_autoencoder_reports_revised\comparison_summary.csv"
$unsupervisedSummaryPath = Join-Path $runDirectory "01_unsupervised_autoencoder_reports_revised\comparison_summary.csv"
$nonlinearSummaryPath = Join-Path $runDirectory "nonlinear_runtime_storage_comparison.csv"

@(
    $pcaSummaryPath,
    $twoStagePath,
    $narrowVariancePath,
    $wideVariancePath,
    $scalarSummaryPath,
    $kpcaSummaryPath,
    $supervisedSummaryPath,
    $unsupervisedSummaryPath,
    $nonlinearSummaryPath
) | ForEach-Object { Assert-FileExists $_ }

# Scientific/protocol checks that catch stale or mismatched result files.
$pcaSummary = @(Import-Csv -LiteralPath $pcaSummaryPath)
if ($pcaSummary.Count -lt 12) {
    throw "PCA summary has $($pcaSummary.Count) rows; at least 12 are expected (3 lengths x 2 protocols x 2 scenarios)."
}
$pcaProtocols = @($pcaSummary | Select-Object -ExpandProperty Cohort_Protocol -Unique)
foreach ($requiredProtocol in @("enrolled_cohort", "heldout_sensor_cohort")) {
    if ($requiredProtocol -notin $pcaProtocols) {
        throw "PCA results are missing protocol: $requiredProtocol"
    }
}

$scalarSummary = @(Import-Csv -LiteralPath $scalarSummaryPath)
if ($scalarSummary.Count -ne 6) {
    throw "DS/PQM summary has $($scalarSummary.Count) rows; exactly 6 natural-length rows are expected."
}
$scalarLengths = @($scalarSummary | Select-Object -ExpandProperty ID_Bits -Unique)
foreach ($requiredLength in @("12", "18", "24")) {
    if ($requiredLength -notin $scalarLengths) {
        throw "DS/PQM results are missing the natural $requiredLength-bit configuration."
    }
}

$nonlinearSummary = @(Import-Csv -LiteralPath $nonlinearSummaryPath)
$linearRows = @($nonlinearSummary | Where-Object { $_.Method -eq "Linear PCA" })
if ($linearRows.Count -lt 4) {
    throw "Consolidated comparison must contain four 128-bit linear-PCA rows (two protocols x two scenarios)."
}
foreach ($row in $linearRows) {
    if ([string]::IsNullOrWhiteSpace($row.Mean_Fit_Seconds) -or
        [string]::IsNullOrWhiteSpace($row.Mean_Transform_Milliseconds)) {
        throw "Linear-PCA runtime fields are blank. The consolidator used stale PCA outputs."
    }
}
$aeAggregateRows = @(
    $nonlinearSummary | Where-Object {
        $_.Row_Type -eq "seed_mean_std" -and
        $_.Method -in @("Supervised autoencoder", "Unsupervised autoencoder")
    }
)
if ($aeAggregateRows.Count -ne 4) {
    throw "Expected four autoencoder mean+standard-deviation rows (2 methods x 2 scenarios)."
}

# Copy compact, manuscript-relevant outputs into one clearly named folder.
Copy-CompactResult $nonlinearSummaryPath $tableDirectory "nonlinear_runtime_storage_comparison.csv"
Copy-CompactResult $pcaSummaryPath $tableDirectory "PCA_comparison_summary.csv"
Copy-CompactResult (Join-Path $runDirectory "01_256reports_revised\fold_results.csv") $tableDirectory "PCA_fold_results.csv"
Copy-CompactResult (Join-Path $runDirectory "01_256reports_revised\per_device_results.csv") $tableDirectory "PCA_per_device_results.csv"
Copy-CompactResult (Join-Path $runDirectory "01_256reports_revised\bit_block_quality.csv") $tableDirectory "PCA_bit_block_quality.csv"
Copy-CompactResult $scalarSummaryPath $tableDirectory "DS_PQM_comparison_summary.csv"
Copy-CompactResult $kpcaSummaryPath $tableDirectory "KPCA_comparison_summary.csv"
Copy-CompactResult $supervisedSummaryPath $tableDirectory "supervised_AE_comparison_summary.csv"
Copy-CompactResult (Join-Path $runDirectory "01_supervised_autoencoder_reports_revised\across_seed_summary.csv") $tableDirectory "supervised_AE_across_seed_summary.csv"
Copy-CompactResult $unsupervisedSummaryPath $tableDirectory "unsupervised_AE_comparison_summary.csv"
Copy-CompactResult (Join-Path $runDirectory "01_unsupervised_autoencoder_reports_revised\across_seed_summary.csv") $tableDirectory "unsupervised_AE_across_seed_summary.csv"

Get-ChildItem -LiteralPath (Join-Path $runDirectory "01_256reports_optimization_revised") -File |
    Where-Object {
        $_.Name -like "optimization_results_*" -or
        $_.Name -like "full_rank_pca_variance_*" -or
        $_.Name -like "pca_variance_*"
    } |
    Copy-Item -Destination $tableDirectory -Force

$reportDirectories = @(
    "01_256reports_revised",
    "01_256reports_optimization_revised",
    "01_scalar_pqm_reports_revised",
    "01_kpca_reports_revised",
    "01_supervised_autoencoder_reports_revised",
    "01_unsupervised_autoencoder_reports_revised"
)
foreach ($reportName in $reportDirectories) {
    $reportPath = Join-Path $runDirectory $reportName
    if (Test-Path -LiteralPath $reportPath -PathType Container) {
        Get-ChildItem -LiteralPath $reportPath -File -Filter "*.png" |
            ForEach-Object {
                $prefixedName = "{0}__{1}" -f $reportName, $_.Name
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $figureDirectory $prefixedName) -Force
            }
    }
}
Copy-Item -Path (Join-Path $logDirectory "*.log") -Destination $consolidatedLogDirectory -Force

$manifest = [ordered]@{
    RunName = Split-Path $runDirectory -Leaf
    StartedAt = $startedAt.ToString("o")
    CompletedAt = $completedAt.ToString("o")
    DurationMinutes = [math]::Round(($completedAt - $startedAt).TotalMinutes, 3)
    PythonExecutable = $pythonExe
    PythonVersion = (& $pythonExe --version 2>&1 | Out-String).Trim()
    DatasetDirectory = $datasetRoot
    SourceDirectory = $sourceRoot
    SelectedScripts = $selectedScripts
    Validation = [ordered]@{
        PCAProtocols = $pcaProtocols
        PCASummaryRows = $pcaSummary.Count
        ScalarBaselineRows = $scalarSummary.Count
        LinearPCAComparisonRows = $linearRows.Count
        AutoencoderAggregateRows = $aeAggregateRows.Count
        Status = "PASS"
    }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $consolidatedDirectory "run_manifest.json") -Encoding UTF8

$readme = @"
This folder contains the compact, manuscript-ready outputs from $($manifest.RunName).

tables\
  Consolidated PCA, KPCA, autoencoder, DS/PQM, variance, and optimization tables.

figures\
  Figures copied from each method's full report directory. Filenames are prefixed
  with the originating report directory to prevent accidental overwriting.

logs\
  Complete console output for every script, in execution order.

run_manifest.json
  Dataset path, Python version, source-file hashes, duration, and validation status.

The complete raw results remain one directory above this folder. The large directed
comparison files are intentionally not duplicated here.
"@
Set-Content -LiteralPath (Join-Path $consolidatedDirectory "README.txt") -Value $readme -Encoding UTF8

Write-Host ""
Write-Host "All analyses completed and validation passed." -ForegroundColor Green
Write-Host "Full run:       $runDirectory"
Write-Host "Compact results: $consolidatedDirectory"