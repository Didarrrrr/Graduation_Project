import numpy as np
from PIL import Image, ImageChops
import cv2
from io import BytesIO


def _threshold_from_sensitivity(sensitivity, base=50):
    try:
        s = float(sensitivity)
    except (TypeError, ValueError):
        s = 0.5
    s = max(0.1, min(1.0, s))
    t = base - (s - 0.5) * 50.0
    return int(max(22, min(68, round(t))))


class ELADetector:
    
    QUALITY_LEVELS = [75, 85, 95]
    
    def __init__(self, quality=95, scale=15, threshold=50, sensitivity=None):
        self.quality = quality
        self.scale = scale
        if sensitivity is not None:
            self.threshold = _threshold_from_sensitivity(sensitivity)
        else:
            self.threshold = threshold
    
    def _compute_ela_difference(self, original, quality):
        resaved_buffer = BytesIO()
        original.save(resaved_buffer, 'JPEG', quality=quality)
        resaved_buffer.seek(0)
        resaved = Image.open(resaved_buffer)
        diff = ImageChops.difference(original, resaved)
        return np.array(diff, dtype=np.float32)
    
    def _multi_quality_ela(self, original):
        diff_maps = []
        for q in self.QUALITY_LEVELS:
            diff_map = self._compute_ela_difference(original, q)
            diff_maps.append(diff_map)
        
        avg_diff = np.mean(diff_maps, axis=0)
        return avg_diff
    
    def _robust_normalize(self, diff_array):
        p99 = np.percentile(diff_array, 99)
        if p99 < 1.0:
            p99 = 1.0
        
        clipped = np.clip(diff_array, 0, p99)
        normalized = (clipped / p99 * 255.0).astype(np.uint8)
        return normalized
    
    def _bilateral_filter_ela(self, ela_gray):
        filtered = cv2.bilateralFilter(ela_gray, d=9, sigmaColor=75, sigmaSpace=75)
        return filtered
    
    def _compute_block_variances(self, gray_diff, block_size=8):
        h, w = gray_diff.shape
        bh = h // block_size
        bw = w // block_size
        
        if bh < 2 or bw < 2:
            return np.array([[np.var(gray_diff)]])
        
        trimmed = gray_diff[:bh * block_size, :bw * block_size]
        
        blocks = trimmed.reshape(bh, block_size, bw, block_size)
        block_vars = np.var(blocks, axis=(1, 3))
        
        return block_vars
    
    def _compute_spatial_heterogeneity(self, block_vars):
        if block_vars.size < 4:
            return 0.0
        
        h, w = block_vars.shape
        if h < 2 or w < 2:
            return 0.0
        
        mean_var = np.mean(block_vars)
        if mean_var < 1e-6:
            return 0.0
        
        padded = np.pad(block_vars, 1, mode='edge')
        
        local_diff = (
            (block_vars - padded[:-2, 1:-1]) ** 2 +  # up
            (block_vars - padded[2:, 1:-1]) ** 2 +    # down
            (block_vars - padded[1:-1, :-2]) ** 2 +   # left
            (block_vars - padded[1:-1, 2:]) ** 2      # right
        )
        
        global_var = np.var(block_vars)
        if global_var < 1e-6:
            return 0.0
        
        heterogeneity = np.mean(local_diff) / (global_var + 1e-6)
        
        return float(heterogeneity)
    
    def analyze(self, image_path):
        try:
            original = Image.open(image_path).convert('RGB')
            
            source_format = ""
            try:
                with Image.open(image_path) as probe:
                    source_format = (probe.format or "").upper()
            except Exception:
                pass
            is_jpeg_source = source_format in ("JPEG", "JPG", "MPO")
            
            avg_diff = self._multi_quality_ela(original)
            
            if len(avg_diff.shape) == 3:
                gray_diff = np.mean(avg_diff, axis=2)
            else:
                gray_diff = avg_diff
            
            normalized_diff = self._robust_normalize(avg_diff)
            
            gray_normalized = cv2.cvtColor(normalized_diff, cv2.COLOR_RGB2GRAY) if len(normalized_diff.shape) == 3 else normalized_diff
            filtered_gray = self._bilateral_filter_ela(gray_normalized)
            
            block_vars = self._compute_block_variances(gray_diff, block_size=8)
            spatial_hetero = self._compute_spatial_heterogeneity(block_vars)
            
            scale_factor = self.scale
            ela_display = np.clip(normalized_diff.astype(np.float32) * (scale_factor / 10.0), 0, 255).astype(np.uint8)
            ela_image = Image.fromarray(ela_display)
            
            ela_array = np.stack([filtered_gray] * 3, axis=-1) if len(filtered_gray.shape) == 2 else filtered_gray
            
            error_metrics = self._calculate_error_metrics(filtered_gray, block_vars)
            
            suspicious_regions = self._detect_suspicious_regions(filtered_gray)
            
            heatmap = self._generate_heatmap(filtered_gray, original.size)
            
            forgery_score = self._calculate_forgery_score(
                error_metrics, suspicious_regions, spatial_hetero
            )
            
            if not is_jpeg_source:
                forgery_score = forgery_score * 0.55
            
            max_diff = float(np.max(avg_diff))
            threshold_exceeded = (max_diff > self.threshold) or (forgery_score >= 50.0)

            return {
                'ela_image': ela_image,
                'heatmap': heatmap,
                'score': forgery_score,
                'error_metrics': error_metrics,
                'suspicious_regions': suspicious_regions,
                'max_diff': max_diff,
                'threshold_exceeded': threshold_exceeded,
                'ela_threshold_used': self.threshold,
                'source_format': source_format,
            }
            
        except Exception as e:
            raise Exception(f"ELA analysis failed: {str(e)}")
    
    def _calculate_error_metrics(self, gray, block_vars=None):
        mean_error = np.mean(gray)
        std_error = np.std(gray)
        max_error = np.max(gray)
        min_error = np.min(gray)
        
        p95 = np.percentile(gray, 95)
        p99 = np.percentile(gray, 99)
        
        high_error_pixels = np.sum(gray > self.threshold)
        total_pixels = gray.size
        high_error_percentage = (high_error_pixels / total_pixels) * 100
        
        metrics = {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'max_error': float(max_error),
            'min_error': float(min_error),
            'percentile_95': float(p95),
            'percentile_99': float(p99),
            'high_error_pixels': int(high_error_pixels),
            'high_error_percentage': float(high_error_percentage),
        }
        
        if block_vars is not None and block_vars.size > 1:
            metrics['block_mean_var'] = float(np.mean(block_vars))
            metrics['block_std_var'] = float(np.std(block_vars))
            block_mean = np.mean(block_vars)
            if block_mean > 1e-6:
                metrics['block_cv'] = float(np.std(block_vars) / block_mean)
            else:
                metrics['block_cv'] = 0.0
        
        return metrics
    
    def _detect_suspicious_regions(self, gray, min_region_size=100):
        gray_u8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
        
        _, binary = cv2.threshold(gray_u8, self.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        height, width = gray.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_region_size:
                x, y, w, h = cv2.boundingRect(contour)
                
                region = gray_u8[y:y+h, x:x+w]
                avg_error = np.mean(region)
                confidence = min(100, (avg_error / 255) * 200)  # Scale to 0-100
                
                regions.append({
                    'x1': x / width,
                    'y1': y / height,
                    'x2': (x + w) / width,
                    'y2': (y + h) / height,
                    'confidence': float(confidence),
                    'area': int(area),
                    'pixel_coords': {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    }
                })
        
        regions.sort(key=lambda r: r['confidence'], reverse=True)
        
        return regions
    
    def _generate_heatmap(self, gray, image_size):
        gray_u8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
        
        blurred = cv2.GaussianBlur(gray_u8, (15, 15), 0)
        
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        heatmap = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_LINEAR)
        
        return Image.fromarray(heatmap)
    
    
    def _calculate_forgery_score(self, error_metrics, suspicious_regions, spatial_hetero=0.0):
        score = 0.0
        
        block_cv = error_metrics.get('block_cv', 0.0)
        if block_cv > 4.0:
            score += 28
        elif block_cv > 3.0:
            score += 18
        elif block_cv > 2.2:
            score += 10
        elif block_cv > 1.6:
            score += 4
        
        if spatial_hetero > 6.0:
            score += 22
        elif spatial_hetero > 4.0:
            score += 13
        elif spatial_hetero > 2.8:
            score += 6
        elif spatial_hetero > 1.8:
            score += 2
        
        high_error_pct = error_metrics['high_error_percentage']
        if high_error_pct > 15:
            score += min(18, high_error_pct * 1.0)
        elif high_error_pct > 5:
            score += high_error_pct * 0.6
        
        high_confidence_regions = [r for r in suspicious_regions if r['confidence'] > 65]
        if len(high_confidence_regions) >= 4:
            avg_confidence = sum(r['confidence'] for r in high_confidence_regions[:5]) / min(len(high_confidence_regions), 5)
            score += min(16, avg_confidence * 0.18)
        elif len(high_confidence_regions) >= 2:
            avg_confidence = sum(r['confidence'] for r in high_confidence_regions[:3]) / min(len(high_confidence_regions), 3)
            score += min(10, avg_confidence * 0.12)
        elif len(high_confidence_regions) == 1:
            score += 4  # Single region is unreliable
        
        p99 = error_metrics.get('percentile_99', 0)
        mean_error = error_metrics.get('mean_error', 0)
        if mean_error > 0:
            ratio = p99 / mean_error
            if ratio > 8:
                score += 12
            elif ratio > 5:
                score += 6
            elif ratio > 3.5:
                score += 2
        
        score = float(min(100.0, 100.0 * (1.0 - np.exp(-score / 58.0))))
        return score


def perform_ela_analysis(image_path, quality=95, sensitivity=None):
    detector = ELADetector(quality=quality, sensitivity=sensitivity)
    return detector.analyze(image_path)