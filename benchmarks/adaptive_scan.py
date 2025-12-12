import os
import numpy as np
import matplotlib.pyplot as plt

from src.dtm_controller import AdaptiveAFMController
from src.vision_evaluator import VisionEvaluator
from src.adaptive_scanner import AdaptiveScanner


def run(grid_x=4, grid_y=1, region_size=300, result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)
    controller = AdaptiveAFMController()
    evaluator = VisionEvaluator()
    scanner = AdaptiveScanner(controller, evaluator, result_dir=result_dir)

    stats = scanner.scan_grid(grid_x=grid_x, grid_y=grid_y, region_size=region_size)
    montage = scanner.save_montage("adaptive_grid.png")
    return stats


if __name__ == "__main__":
    stats = run()
    print("Adaptive:", stats)