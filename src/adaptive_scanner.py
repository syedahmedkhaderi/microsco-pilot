'''Adaptive scanning algorithm with ML integration'''
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

# Try to import ML predictor (optional - falls back to rules if not available)
try:
    from src.ml_predictor import MLPredictor
    HAS_ML = True
except ImportError:
    HAS_ML = False


class AdaptiveScanner:
    '''Real-time adaptive AFM scanning with ML + rule-based hybrid system'''

    def __init__(self, use_ml=True):
        """Initialize scanner
        
        Args:
            use_ml: Whether to use ML predictions (True) or only rules (False)
        """
        self.controller = AdaptiveAFMController()
        self.evaluator = VisionEvaluator()
        
        # ML predictor (hybrid approach)
        self.use_ml = use_ml and HAS_ML
        if self.use_ml:
            try:
                self.ml_predictor = MLPredictor()
                print("🧠 ML predictor loaded")
            except Exception as e:
                print(f"⚠️  ML predictor failed to load: {e}")
                self.use_ml = False
                self.ml_predictor = None
        else:
            self.ml_predictor = None
        
        self.results = {
            'images': [],
            'analyses': [],
            'time_log': [],
            'param_log': [],
            'ml_used': []  # Track when ML was used
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
            
            # Get real quality from physics (if available)
            real_quality = None
            if not self.controller.simulation_mode:
                real_quality = self.controller.get_quality_from_scan(self.controller.last_scan_result)

            # HYBRID APPROACH: Try ML first, fallback to rules
            ml_used = False
            
            if adaptive and self.use_ml and self.ml_predictor is not None:
                # Try ML prediction
                ml_prediction = self.ml_predictor.predict(image)
                
                if ml_prediction['use_ml']:
                    # ML is confident - use its predictions
                    suggested = ml_prediction['parameters']
                    ml_used = True
                else:
                    # ML not confident - fall back to rules
                    analysis = self.evaluator.analyze_region(
                        image, 
                        current_params=self.controller.scan_params,
                        real_quality=real_quality
                    )
                    suggested = analysis.get('suggested_params')
            else:
                # No ML or not adaptive - use rule-based approach
                analysis = self.evaluator.analyze_region(
                    image, 
                    current_params=self.controller.scan_params,
                    real_quality=real_quality
                )
                suggested = analysis.get('suggested_params')
            
            # For consistency, always get analysis (needed for quality score)
            if 'analysis' not in locals():
                analysis = self.evaluator.analyze_region(
                    image, 
                    current_params=self.controller.scan_params,
                    real_quality=real_quality
                )
                
            # ONLINE LEARNING: Update ML model with real experience
            if self.use_ml and self.ml_predictor is not None:
                # We feed the current image, the parameters used, and the RESULTING quality
                # This allows the ML to learn "Action -> Reward" mapping
                self.ml_predictor.update_online(
                    image,
                    self.controller.scan_params,
                    analysis['quality']
                )

            # Determine if we should adjust parameters
            current = self.controller.scan_params
            should_adjust = False
            if adaptive and suggested is not None:
                # Check if suggested differs from current
                tol_speed = 0.05 * max(1.0, current.get('speed', 1.0))
                tol_force = 0.05 * max(0.5, current.get('force', 0.5))
                tol_res = 1
                if (
                    abs(suggested.get('speed', current.get('speed', 0)) - current.get('speed', 0)) > tol_speed or
                    abs(suggested.get('force', current.get('force', 0)) - current.get('force', 0)) > tol_force or
                    abs(int(suggested.get('resolution', current.get('resolution', 0)) - int(current.get('resolution', 0)))) > tol_res
                ):
                    should_adjust = True

            # Compose analysis fields
            analysis['should_adjust'] = should_adjust
            analysis['recommendations'] = suggested if suggested is not None else {}
            analysis['ml_used'] = ml_used

            quality_scores.append(analysis['quality'])
            total_time += scan_time

            # Print with ML indicator
            ml_indicator = " 🧠" if ml_used else ""
            print(f"   Region {i+1}/{num_regions}: "
                  f"Quality={analysis['quality']:.1f}, "
                  f"Complexity={analysis['complexity']:.1f}, "
                  f"Time={scan_time:.1f}s{ml_indicator}")

            # Adapt parameters if enabled
            if adaptive and analysis['should_adjust']:
                self.controller.update_params(analysis['recommendations'])

            # Store results
            self.results['images'].append(image)
            self.results['analyses'].append(analysis)
            self.results['time_log'].append(scan_time)
            self.results['param_log'].append(self.controller.scan_params.copy())
            self.results['ml_used'].append(ml_used)

        # Summary
        avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
        print(f"\n📊 Scan complete:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average quality: {avg_quality:.1f}/10")
        print(f"   Quality std: {float(np.std(quality_scores)):.1f}")
        
        # Show ML usage if enabled
        if self.use_ml:
            ml_count = sum(self.results['ml_used'])
            print(f"   ML used: {ml_count}/{num_regions} times 🧠")

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