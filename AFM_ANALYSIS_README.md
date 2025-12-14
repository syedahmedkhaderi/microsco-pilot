# SmartScan - Real AFM Data Analysis

## What You Have

You now have **16 real AFM h5 files** from the DTMicroscope repository in `data/AFM/`:
- 3 PMN28Pt samples
- 7 PTO_Funakubo samples  
- 6 PZT samples

## How to Run

### Method 1: Analyze ALL AFM files at once (Recommended)
```bash
python analyze_afm.py
```
This will:
- Process all 16 h5 files automatically
- Extract 10 regions from each
- Analyze quality, complexity, and sharpness
- Generate visualizations for each file
- Save results to `results/afm_visualizations/`

### Method 2: Analyze a SPECIFIC file
```bash
python analyze_afm.py --file data/AFM/PMN28Pt0000.h5
```

### Method 3: Inspect file structure first
```bash
python analyze_afm.py --inspect data/AFM/PMN28Pt0000.h5
```
This shows you the internal h5 file structure before analyzing.

### Method 4: Analyze with more regions
```bash
python analyze_afm.py --regions 20
```
Extracts and analyzes 20 regions instead of default 10.

### Method 5: Run the original SmartScan benchmark
```bash
python main.py
```
This runs the SmartScan demo with synthetic data (for comparison/benchmarking).

## What Happens When You Run

When you run `python analyze_afm.py`, here's what happens:

1. **Loads each h5 file** using the h5_data_loader module
2. **Extracts metadata** (AFM mode, resolution, scan size)
3. **Generates visualization** of the raw AFM topography
4. **Divides into regions** (default 10 regions per file)
5. **Analyzes each region** with SmartScan's VisionEvaluator
   - Quality score (image clarity)
   - Complexity score (structural detail)
   - Sharpness score (edge definition)
6. **Shows summary** with average scores
7. **Saves visualizations** to results folder

## Output

You'll get:
```
======================================================================
  SMARTSCAN - REAL AFM DATA ANALYSIS
  Found 16 AFM files
======================================================================

  ANALYZING: PMN28Pt0000.h5
----------------------------------------------------------------------

📊 Dataset Information:
   AFM Mode: contact
   Resolution: 256 px
   Scan Size: 10 um

🎨 Generating visualization...
💾 Saved visualization to: results/afm_visualizations/PMN28Pt0000_visualization.png

🔬 Extracting 10 scan regions...

🤖 Analyzing with SmartScan...
   Region  1/10: Quality= 5.2/10, Complexity= 2.1/10, Sharpness= 4.8/10
   Region  2/10: Quality= 6.1/10, Complexity= 1.8/10, Sharpness= 5.3/10
   ...

📈 Summary:
   Average Quality:    5.67/10
   Average Complexity: 2.03/10
   Average Sharpness:  5.12/10
   Visualization: results/afm_visualizations/PMN28Pt0000_visualization.png

[... repeats for all 16 files ...]

======================================================================
  OVERALL SUMMARY
======================================================================
   PMN28Pt0000.h5                 Quality: 5.67/10
   PMN28Pt0001.h5                 Quality: 5.82/10
   ...
   PZT_thick0009.h5               Quality: 6.23/10

✅ Analysis complete!
   Visualizations saved to: results/afm_visualizations/
```

## Files You Have Now

```
smartScan/
├── analyze_afm.py          ← NEW! Simple script to analyze your AFM data
├── main.py                 ← Original SmartScan benchmark
├── src/
│   ├── h5_data_loader.py   ← Module for loading h5 files
│   ├── adaptive_scanner.py
│   └── vision_evaluator.py
├── data/
│   └── AFM/               ← YOUR 16 REAL AFM FILES
│       ├── PMN28Pt0000.h5
│       ├── PMN28Pt0001.h5
│       ├── ... (14 more)
│       └── PZT_thick0009.h5
└── results/
    └── afm_visualizations/ ← Generated AFM images go here
```

## What We Removed

❌ Deleted (not needed anymore):
- `examples/h5_loader_examples.py` - Was for synthetic data testing
- `examples/afm_integration_demo.py` - Replaced with simpler `analyze_afm.py`
- `data/synthetic_afm.h5` - You have real data now
- `docs/AFM_DATA_LOADING.md` - Replaced with this file
- `docs/H5_LOADER_FAQ.md` - Replaced with this file

✅ Kept (you still need these):
- `src/h5_data_loader.py` - Core module for loading h5 files
- `main.py` - Original SmartScan benchmark demo
- All your real AFM h5 files in `data/AFM/`

## Quick Start

Just run:
```bash
python analyze_afm.py
```

That's it! It will analyze all your AFM files automatically.
