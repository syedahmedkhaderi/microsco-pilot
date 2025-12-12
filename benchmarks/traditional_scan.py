import os
import numpy as np
import matplotlib.pyplot as plt

from src.dtm_controller import AdaptiveAFMController
from src.vision_evaluator import VisionEvaluator


def run(grid_x=4, grid_y=1, region_size=300, result_dir="results", base_params=None):
    os.makedirs(result_dir, exist_ok=True)
    controller = AdaptiveAFMController()
    evaluator = VisionEvaluator()

    base = base_params or {'speed': 5.0, 'force': 1.0, 'resolution': 256}
    controller.update_params(base)

    records = []
    max_x = max(controller.sample_size - region_size, 0)
    max_y = max(controller.sample_size - region_size, 0)
    xs = np.linspace(0, max_x, grid_x).astype(int)
    ys = np.linspace(0, min(max_y, region_size * max(0, grid_y - 1)), grid_y).astype(int)

    for y in ys:
        for x in xs:
            img, t = controller.scan_region(int(x), int(y), int(region_size))
            analysis = evaluator.analyze_region(img, current_params=controller.scan_params)
            fpath = os.path.join(result_dir, f"baseline_x{x}_y{y}.png")
            try:
                img.save(fpath)
            except Exception:
                pass
            records.append({
                'x': int(x), 'y': int(y), 'time': float(t),
                'quality': float(analysis['quality']),
                'path': fpath,
            })

    total_time = sum(r['time'] for r in records)
    avg_quality = float(np.mean([r['quality'] for r in records]))

    # Montage
    try:
        import PIL
        imgs = [PIL.Image.open(r['path']) for r in records if os.path.exists(r['path'])]
        cols = int(np.ceil(np.sqrt(len(imgs))))
        rows = int(np.ceil(len(imgs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = np.array(axes).reshape(rows, cols)
        for idx, im in enumerate(imgs):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.imshow(im, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        for idx in range(len(imgs), rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis('off')
        plt.tight_layout()
        outpath = os.path.join(result_dir, "baseline_grid.png")
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {
        'regions': len(records),
        'total_time': total_time,
        'avg_quality': avg_quality,
    }


if __name__ == "__main__":
    stats = run()
    print("Baseline:", stats)