param(
    [switch]$Clean,
    [switch]$CliOnly,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')"
    }
}

if (-not $SkipTests) {
    Invoke-Checked "python" @("-m", "pytest", "-q")
}

$buildDir = Join-Path $root "build"
$distDir = Join-Path $root "dist"
# PyInstaller output is disposable project-local build state.  Cleaning it on
# every release build prevents a stale one-file artifact from colliding with
# the onedir bundle produced by the spec.
if (Test-Path $buildDir) { Remove-Item -LiteralPath $buildDir -Recurse -Force }
if (Test-Path $distDir) { Remove-Item -LiteralPath $distDir -Recurse -Force }

if ($CliOnly) {
    # CLI-only packaging is a fallback for environments without PySide6.
    $entry = Join-Path $root "visual_continuity_lab\__main__.py"
    Invoke-Checked "python" @(
        "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", "VisualContinuityLab", "--paths", $root, $entry
    )
} else {
    Invoke-Checked "python" @("-m", "PyInstaller", "--noconfirm", "--clean", "visual_continuity_lab.spec")
}

$bundle = Join-Path $distDir "VisualContinuityLab"
if (-not (Test-Path (Join-Path $bundle "VisualContinuityLab.exe"))) {
    throw "PyInstaller completed but VisualContinuityLab.exe was not found in $bundle"
}

foreach ($item in @("README.md", "LICENSE", "docs", "schemas", "demo")) {
    $source = Join-Path $root $item
    $target = Join-Path $bundle $item
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination $target -Recurse -Force
    }
}

$version = "0.1.0"
$archive = Join-Path $distDir ("VisualContinuityLab-{0}-win64.zip" -f $version)
if (Test-Path $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path $bundle -DestinationPath $archive
Write-Host "Built: $bundle\VisualContinuityLab.exe"
Write-Host "Archive: $archive"
