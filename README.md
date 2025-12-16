# Microscopy Hackathon: Team SmartScan

**Syed, Ahmed and Ali** - University of Doha for Science & Technology

## Background
Atomic force microscopy (AFM) scans are notoriously slow. A typical high-quality scan can take hours, and operators often set conservative "safe" parameters (slow speed) to avoid poor tracking or thermal drift errors. This "one-size-fits-all" approach wastes massive amounts of time on flat regions and fails to adapt to complex features, resulting in suboptimal data throughput.

## Introduction
SmartScan reimagines AFM scanning as a dynamic, **Physics-Informed Machine Learning** problem. Instead of using static parameters, SmartScan uses an ML model trained on **real AFM data** and **physics simulations** to continuously predict the optimal scan parameters (speed, resolution, force) for every region of the sample.

Unlike traditional methods that rely on fixed rules or simple computer vision, SmartScan's ML engine has learned the complex physical relationship between **scan speed** and **image quality**, specifically understanding the trade-off between **thermal drift** (which ruins slow scans) and **tracking error** (which ruins fast scans). It identifies the "sweet spot" in real-time.

Key innovations:
*   **Physics-Informed ML:** Trained on 160+ real AFM regions with simulated physics artifacts (Drift, Noise, PID lag).
*   **Real-Time Adaptation:** Dynamically adjusts speed from 1 µm/s to 20 µm/s based on local surface complexity.
*   **Drift-Aware Control:** Penalizes overly slow scans to minimize thermal drift accumulation, a common issue in long experiments.

## Code
```bash
# Clone the repository and install dependencies
pip install -r requirements.txt

# Run the benchmark (uses Real AFM Data from data/AFM/)
python main.py

# Outputs in ./results/
# - smartscan_real_data.png (Benchmark Plot)
# - detailed logs and metrics
```

## Results
*Results based on benchmark of 10 real AFM regions from the PZT/PMN dataset.*

<p align="center">
  <img src="./results/smartscan_real_data.png" width="90%" />
  <br/>
  <em>Figure 1: Benchmark Results. SmartScan (Green) consistently achieves higher quality in less time compared to Traditional (Red).</em>
</p>

| Metric            | Traditional (Fixed) | SmartScan (Adaptive) | Improvement |
|-------------------|---------------------|----------------------|-------------|
| Total scan time   | 20,531 s (~5.7 hr)  | 13,016 s (~3.6 hr)   | **36.6% FASTER** |
| Average quality   | 8.45 / 10           | 8.82 / 10            | **+0.37 POINTS** |
| ML Usage          | N/A                 | 100%                 | Full Automation  |

**Performance Analysis:**
1.  **Speed (37% Faster):** Traditional scanning is stuck at a "safe" 5.0 µm/s. SmartScan intelligently identifies that many regions can be scanned at **10.5 µm/s** or higher without losing fidelity, cutting hours off the experiment time.
2.  **Quality (Superior):** Counter-intuitively, SmartScan's faster speed *improves* quality. By scanning faster, SmartScan minimizes the impact of **thermal drift** (random noise accumulation over time), which degrades the slow Traditional scans. The ML model optimized this Drift-vs-Error trade-off perfectly.
3.  **Real Data Validation:** Verified on **actual PZT and PMN thin film AFM data**, ensuring the system works on real-world scientific samples, not just synthetic tests.

## Methods
**Physics-Informed Training:**
We trained a LightGBM regressor on a dataset of **160 real AFM regions**. For each region, we simulated thousands of scans using the **DTMicroscope** physics engine, injecting realistic **thermal drift noise** and **PID tracking errors** to create a ground-truth map of "Parameters → Quality". The model learned to predict the parameters that maximize quality (minimizing both drift and tracking error).

**Drift Simulation:**
A key component of our success was correctly modeling **Thermal Drift**. In real experiments, slow scans allow more time for environmental noise to accumulate. We integrated a drift penalty into the DTMicroscope simulation path (`drift_noise ~ 1/speed`), creating a realistic physical incentive for the AI to scan efficiently.

**Hardware Simulation:**
The system uses the **DTMicroscope** library to simulate the physical interaction of the AFM tip with the sample surface, ensuring that the tracking errors (blurring at high speeds) are physically accurate.

## Conclusions
SmartScan proves that **Machine Learning** can outperform human operators by solving the complex multi-variable optimization problem of AFM scanning. By balancing thermal drift against tracking error, it achieved a **37% speedup** and **superior image quality** on real data. This capability transforms the microscope from a passive tool into an intelligent agent, capable of autonomous, high-throughput materials characterization.