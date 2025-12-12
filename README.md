# SmartScan

Real-time adaptive AFM scanning using vision-based feedback. Built for the December 2025 Microscopy Hackathon. Focused, small, demo-ready in a single day.

## Features
- Uses DTMicroscope AFM simulator when available, else deterministic fallback.
- Vision evaluator computes sharpness, complexity, and noise for parameter tuning.
- Adaptive scanner adjusts speed, force, and resolution per-region.
- Benchmarks compare baseline vs adaptive (time and quality).

## Project Structure
```
SmartScan/
  src/
    dtm_controller.py       # DTMicroscope wrapper and fallback simulator
    vision_evaluator.py     # Fast CV quality assessment + recommendations
    adaptive_scanner.py     # Orchestrates adaptive scanning
  benchmarks/
    traditional_scan.py     # Fixed-parameter baseline
    adaptive_scan.py        # Adaptive method
  results/                  # Auto-generated plots and images
  main.py                   # Demo entrypoint
  requirements.txt
  README.md
```

## Install
```
pip install -r requirements.txt
```

## Run Demo (≈2 minutes)
```
python main.py --grid_x 4 --grid_y 1 --region_size 300
```
Outputs are saved to `results/`:
- `baseline_grid.png`, `adaptive_grid.png`: montages of scanned regions
- `benchmark_comparison.png`: time vs quality comparison

## How It Works
- `AdaptiveAFMController` exposes `scan_region` and `update_params`. If DTMicroscope is unavailable, a realistic synthetic topography is generated with speed-dependent noise.
- `VisionEvaluator` measures sharpness (Laplacian variance), complexity (Sobel gradient statistics), and noise (high-frequency residual). It recommends parameter adjustments.
- `AdaptiveScanner` iterates through regions, analyzes images, updates parameters, and records outputs.

## Benchmarks
- Baseline: fixed speed/force/resolution across the grid.
- Adaptive: parameters updated per-region according to image content.
- The comparison plot shows time savings or quality improvements depending on sample features.

## Notes
- Parameter bounds are enforced: resolution [128, 512], speed [1, 20] µm/s, force [0.5, 5] nN.
- Adjust thresholds and grid sizes to trade speed vs quality for your demo.