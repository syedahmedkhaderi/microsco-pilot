'''Vision-based quality evaluation for AFM images'''
import numpy as np
from PIL import Image

try:
    from scipy import ndimage as nd_image
except Exception:
    nd_image = None


class VisionEvaluator:
    '''Fast computer vision-based image quality assessment'''

    def analyze_region(self, image, current_params=None, real_quality=None):
        '''Analyze AFM image and recommend parameters

        Args:
            image: PIL Image or numpy array
            current_params: optional dict of current scan params
            real_quality: optional dict of real quality metrics from DTMicroscope

        Returns:
            analysis: dict with quality, complexity, recommendations, suggested_params
        '''
        if isinstance(image, Image.Image):
            img = np.array(image).astype(float)
        else:
            img = image.astype(float)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        grad_mag = self._gradient_magnitude(img)
        lap_var = self._laplacian_variance(img)
        noise_level = self._noise_level(img)
        contrast = float(img.std())

        complexity = float(grad_mag.std())
        
        if real_quality is not None:
            # Use REAL quality from physics simulation
            quality = float(real_quality['quality'])
        else:
            # Fall back to CV-based estimate
            quality = float((lap_var + contrast) / (noise_level + 1e-6))

        recommendations = []

        if complexity / (complexity + 1) > 0.6:
            recommendations.append("increase resolution, reduce speed for fine features")
        if noise_level / (noise_level + 1) > 0.6:
            recommendations.append("reduce speed to lower noise; adjust force slightly")
        if (quality / (quality + 1) > 0.7) and (complexity / (complexity + 1) < 0.4):
            recommendations.append("increase speed; lower resolution to save time")
        if not recommendations:
            recommendations.append("maintain parameters; minor adjustments only")

        suggested_params = None
        if current_params is not None:
            suggested_params = self._suggest_params(current_params, quality, complexity, noise_level)

        return {
            'quality': quality,
            'complexity': complexity,
            'noise': float(noise_level),
            'sharpness': float(lap_var),
            'contrast': float(contrast),
            'recommendations': recommendations,
            'suggested_params': suggested_params,
        }

    def _gradient_magnitude(self, img):
        if nd_image is not None:
            gx = nd_image.sobel(img, axis=1, mode='reflect')
            gy = nd_image.sobel(img, axis=0, mode='reflect')
            return np.hypot(gx, gy)
        # Fallback Sobel
        sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
        sy = sx.T
        gx = self._convolve2d(img, sx)
        gy = self._convolve2d(img, sy)
        return np.hypot(gx, gy)

    def _laplacian_variance(self, img):
        if nd_image is not None:
            lap = nd_image.laplace(img, mode='reflect')
        else:
            k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            lap = self._convolve2d(img, k)
        return float(lap.var())

    def _noise_level(self, img):
        if nd_image is not None:
            low = nd_image.gaussian_filter(img, sigma=1.0, mode='reflect')
        else:
            k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
            k /= k.sum()
            low = self._convolve2d(img, k)
        high = img - low
        return float(high.std())

    def _convolve2d(self, img, kernel):
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out = np.zeros_like(img)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                region = padded[i:i + kh, j:j + kw]
                out[i, j] = np.sum(region * kernel)
        return out

    def _suggest_params(self, current, quality, complexity, noise):
        speed = float(current.get('speed', 5.0))
        force = float(current.get('force', 1.0))
        res = int(current.get('resolution', 256))

        comp_n = complexity / (complexity + 1)
        noise_n = noise / (noise + 1)
        qual_n = quality / (quality + 1)

        if comp_n > 0.6:
            res *= 1.5
            speed *= 0.7
            force *= 1.1
        elif noise_n > 0.6:
            speed *= 0.6
            force *= 0.9
        elif qual_n > 0.7 and comp_n < 0.4:
            speed *= 1.3
            res *= 0.8
            force *= 1.0
        else:
            speed *= 1.0
            res *= 1.0
            force *= 1.0

        res = int(np.clip(res, 128, 512))
        speed = float(np.clip(speed, 1.0, 20.0))
        force = float(np.clip(force, 0.5, 5.0))

        return {
            'speed': speed,
            'force': force,
            'resolution': res,
        }