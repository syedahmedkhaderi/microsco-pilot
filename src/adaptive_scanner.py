import os
import numpy as np

from .dtm_controller import AdaptiveAFMController
from .vision_evaluator import VisionEvaluator


class AdaptiveScanner:
    def __init__(self, controller: AdaptiveAFMController, evaluator: VisionEvaluator, result_dir: str = "results"):
        self.controller = controller
        self.evaluator = evaluator
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.records = []

    def scan_grid(self, grid_x=4, grid_y=1, region_size=300, start=(0, 0)):
        sx, sy = start
        max_x = max(self.controller.sample_size - region_size, 0)
        max_y = max(self.controller.sample_size - region_size, 0)
        xs = np.linspace(sx, max_x, grid_x).astype(int)
        ys = np.linspace(sy, min(max_y, sy + region_size * max(0, grid_y - 1)), grid_y).astype(int)

        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                img, t = self.controller.scan_region(int(x), int(y), int(region_size))
                analysis = self.evaluator.analyze_region(img, current_params=self.controller.scan_params)
                suggested = analysis.get('suggested_params') or {}
                if suggested:
                    self.controller.update_params(suggested)

                fname = os.path.join(self.result_dir, f"adaptive_x{x}_y{y}.png")
                try:
                    img.save(fname)
                except Exception:
                    pass

                self.records.append({
                    'x': int(x), 'y': int(y), 'time': float(t),
                    'quality': float(analysis['quality']),
                    'complexity': float(analysis['complexity']),
                    'params': self.controller.scan_params.copy(),
                    'path': fname,
                })

        return self.summary()

    def summary(self):
        if not self.records:
            return {}
        total_time = sum(r['time'] for r in self.records)
        avg_quality = float(np.mean([r['quality'] for r in self.records]))
        avg_speed = float(np.mean([r['params']['speed'] for r in self.records]))
        return {
            'regions': len(self.records),
            'total_time': total_time,
            'avg_quality': avg_quality,
            'avg_speed': avg_speed,
        }

    def save_montage(self, outfile="adaptive_grid.png"):
        import matplotlib.pyplot as plt
        import PIL
        thumbs = [r['path'] for r in self.records if r.get('path') and os.path.exists(r['path'])]
        if not thumbs:
            return None
        images = [PIL.Image.open(p) for p in thumbs]
        cols = int(np.ceil(np.sqrt(len(images))))
        rows = int(np.ceil(len(images) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = np.array(axes).reshape(rows, cols)
        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        for idx in range(len(images), rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis('off')
        plt.tight_layout()
        outpath = os.path.join(self.result_dir, outfile)
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        return outpath