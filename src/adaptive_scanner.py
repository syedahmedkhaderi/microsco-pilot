'''Adaptive scanning algorithm'''
import os
import sys
import time
import numpy as np

# Ensure project root on path when running as a script: python src/adaptive_scanner.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dtm_controller import AdaptiveAFMController
from src.vision_evaluator import VisionEvaluator


class AdaptiveScanner:
    '''Real-time adaptive AFM scanning'''

    def __init__(self):
        self.controller = AdaptiveAFMController()
        self.evaluator = VisionEvaluator()
        self.results = {
            'images': [],
            'analyses': [],
            'time_log': [],
            'param_log': []
        }

    def scan_sample(self, num_regions=10, adaptive=True):
        '''Scan sample with or without adaptation

        Args:
            num_regions: Number of regions to scan
            adaptive: Enable adaptive parameter adjustment

        Returns:
            results: Dict with all scan data
        '''
        print(f"\n🔬 Starting {'ADAPTIVE' if adaptive else 'TRADITIONAL'} scan...")
        print(f"   Scanning {num_regions} regions\n")

        total_time = 0
        quality_scores = []

        for i in range(num_regions):
            # Calculate position (scan left to right, top to bottom)
            x = (i % 5) * 200  # 5 regions across
            y = (i // 5) * 200  # Move down after each row

            # Scan region
            start_time = time.time()
            image, scan_time = self.controller.scan_region(x, y, size=100)

            # Analyze (provide current params so evaluator can suggest updates)
            analysis = self.evaluator.analyze_region(image, current_params=self.controller.scan_params)

            # Harmonize fields expected by this algorithm
            suggested = analysis.get('suggested_params')
            current = self.controller.scan_params
            should_adjust = False
            if adaptive and suggested is not None:
                # Determine if suggested differs from current beyond small tolerance
                tol_speed = 0.05 * max(1.0, current.get('speed', 1.0))
                tol_force = 0.05 * max(0.5, current.get('force', 0.5))
                tol_res = 1
                if (
                    abs(suggested.get('speed', current.get('speed', 0)) - current.get('speed', 0)) > tol_speed or
                    abs(suggested.get('force', current.get('force', 0)) - current.get('force', 0)) > tol_force or
                    abs(int(suggested.get('resolution', current.get('resolution', 0)) - int(current.get('resolution', 0)))) > tol_res
                ):
                    should_adjust = True

            # Compose analysis fields for downstream logging
            analysis['should_adjust'] = should_adjust
            analysis['recommendations'] = suggested if suggested is not None else {}

            quality_scores.append(analysis['quality'])
            total_time += scan_time

            print(f"   Region {i+1}/{num_regions}: "
                  f"Quality={analysis['quality']:.1f}, "
                  f"Complexity={analysis['complexity']:.1f}, "
                  f"Time={scan_time:.1f}s")

            # Adapt parameters if enabled
            if adaptive and analysis['should_adjust']:
                self.controller.update_params(analysis['recommendations'])

            # Store results
            self.results['images'].append(image)
            self.results['analyses'].append(analysis)
            self.results['time_log'].append(scan_time)
            self.results['param_log'].append(self.controller.scan_params.copy())

        # Summary
        avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
        print(f"\n📊 Scan complete:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average quality: {avg_quality:.1f}/10")
        print(f"   Quality std: {float(np.std(quality_scores)):.1f}")

        self.results['total_time'] = total_time
        self.results['avg_quality'] = avg_quality
        self.results['mode'] = 'adaptive' if adaptive else 'traditional'

        return self.results


def compare_methods(num_regions=10):
    '''Compare traditional vs adaptive scanning'''
    print("="*60)
    print("SMARTSCAN BENCHMARK")
    print("="*60)

    # Traditional scan (fixed parameters)
    print("\n1️⃣  TRADITIONAL SCANNING")
    scanner_trad = AdaptiveScanner()
    results_trad = scanner_trad.scan_sample(num_regions, adaptive=False)

    # Adaptive scan
    print("\n2️⃣  ADAPTIVE SCANNING (SmartScan)")
    scanner_adapt = AdaptiveScanner()
    results_adapt = scanner_adapt.scan_sample(num_regions, adaptive=True)

    # Calculate improvements
    time_saved = results_trad['total_time'] - results_adapt['total_time']
    time_improvement = (time_saved / results_trad['total_time']) * 100 if results_trad['total_time'] > 0 else 0.0

    quality_gain = results_adapt['avg_quality'] - results_trad['avg_quality']

    print("\n" + "="*60)
    print("🏆 RESULTS")
    print("="*60)
    print(f"\n⏱️  Time:")
    print(f"   Traditional: {results_trad['total_time']:.1f}s")
    print(f"   SmartScan:   {results_adapt['total_time']:.1f}s")
    print(f"   ✅ Saved {time_saved:.1f}s ({time_improvement:.1f}% faster)")

    print(f"\n📊 Quality:")
    print(f"   Traditional: {results_trad['avg_quality']:.2f}/10")
    print(f"   SmartScan:   {results_adapt['avg_quality']:.2f}/10")
    if quality_gain > 0:
        print(f"   ✅ Improved by {quality_gain:.2f} points")

    return results_trad, results_adapt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SmartScan adaptive vs traditional benchmark")
    parser.add_argument("--regions", type=int, default=10, help="Number of regions to scan")
    args = parser.parse_args()
    compare_methods(num_regions=args.regions)