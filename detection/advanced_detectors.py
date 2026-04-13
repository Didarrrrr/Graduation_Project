import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kurtosis as scipy_kurtosis
from skimage.restoration import estimate_sigma


def _zigzag_indices(n, k):
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r_start = min(s, n - 1)
            c_start = s - r_start
            while r_start >= 0 and c_start < n:
                indices.append((r_start, c_start))
                r_start -= 1
                c_start += 1
        else:
            c_start = min(s, n - 1)
            r_start = s - c_start
            while c_start >= 0 and r_start < n:
                indices.append((r_start, c_start))
                r_start += 1
                c_start -= 1
    ac_indices = [(r, c) for r, c in indices if not (r == 0 and c == 0)]
    return ac_indices[:k]


class CopyMoveDetector:

    N_COEFFS = 16
    QUANT_STEP = 100.0
    MATCH_RADIUS = 0.07
    MIN_SPATIAL_DIST_FACTOR = 3.0
    RANSAC_INLIER_THRESHOLD = 4.0
    RANSAC_MIN_INLIERS = 12

    def __init__(self, block_size=16, step=8):
        self.block_size = block_size
        self.step = step
        self._zigzag_16 = _zigzag_indices(16, self.N_COEFFS)
        self._zigzag_8 = _zigzag_indices(8, self.N_COEFFS)

    def analyze(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to load image for copy-move analysis")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        all_pairs = []
        total_blocks = 0

        for bsz in (8, 16):
            step = max(4, bsz // 2)
            zigzag = self._zigzag_8 if bsz <= 8 else self._zigzag_16
            blocks = self._extract_blocks(gray, bsz, step, zigzag)
            total_blocks += len(blocks)

            if len(blocks) < 30:
                continue

            features = np.array([b["feature"] for b in blocks], dtype=np.float32)
            positions = np.array([b["pos"] for b in blocks], dtype=np.float32)
            pairs = self._match_kdtree(features, positions, bsz)

            if pairs:
                verified = self._ransac_verify(pairs)
                all_pairs.extend(verified)

        if not all_pairs:
            return {
                "score": 0.0,
                "regions": [],
                "block_count": total_blocks,
                "match_count": 0,
            }

        displacement_vectors = np.array([p2 - p1 for p1, p2, _ in all_pairs], dtype=np.float32)
        labels = self._cluster_displacements(displacement_vectors)

        regions = self._build_regions(all_pairs, labels, w, h)
        dominant_cluster_size = max((r["support_matches"] for r in regions), default=0)
        match_count = len(all_pairs)

        dominant_ratio = dominant_cluster_size / max(1, match_count)
        total_region_area = sum(r["area"] for r in regions)
        coverage_ratio = min(1.0, total_region_area / float(max(1, w * h)))

        score = (
            (dominant_ratio * 45.0)
            + min(25.0, max(0.0, dominant_cluster_size - 8) * 1.6)
            + min(15.0, coverage_ratio * 100.0)
        )
        score = min(100.0, score)

        return {
            "score": round(float(score), 2),
            "regions": regions,
            "block_count": total_blocks,
            "match_count": match_count,
        }

    def _extract_blocks(self, gray, block_size, step, zigzag_indices):
        gray_f = gray.astype(np.float32) / 255.0
        h, w = gray_f.shape
        raw_blocks = []

        for y in range(0, h - block_size + 1, step):
            for x in range(0, w - block_size + 1, step):
                block = gray_f[y : y + block_size, x : x + block_size]
                dct = cv2.dct(block)

                n_avail = min(len(zigzag_indices), self.N_COEFFS)
                signature = np.array(
                    [dct[r, c] for r, c in zigzag_indices[:n_avail]],
                    dtype=np.float32,
                )

                if len(signature) < self.N_COEFFS:
                    signature = np.pad(signature, (0, self.N_COEFFS - len(signature)))

                nrm = float(np.linalg.norm(signature))
                if nrm > 1e-6:
                    signature = signature / nrm

                quantized = np.round(signature * self.QUANT_STEP) / self.QUANT_STEP
                variance = float(np.var(block))
                raw_blocks.append(
                    {
                        "feature": quantized,
                        "pos": np.array([x, y], dtype=np.float32),
                        "variance": variance,
                    }
                )

        if not raw_blocks:
            return []

        variances = np.array([b["variance"] for b in raw_blocks], dtype=np.float64)
        var_threshold = self._otsu_variance_threshold(variances)

        blocks = [b for b in raw_blocks if b["variance"] >= var_threshold]
        return blocks

    def _otsu_variance_threshold(self, variances):
        if len(variances) < 10:
            return max(0.0015, float(np.percentile(variances, 35)))

        v_min, v_max = np.min(variances), np.max(variances)
        if v_max - v_min < 1e-8:
            return float(v_min)

        normalized = ((variances - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        threshold = v_min + (otsu_val / 255.0) * (v_max - v_min)

        return max(0.001, float(threshold))

    def _match_kdtree(self, features, positions, block_size):
        tree = cKDTree(features)
        pairs = []
        min_spatial = block_size * self.MIN_SPATIAL_DIST_FACTOR

        pair_indices = tree.query_pairs(r=self.MATCH_RADIUS, output_type='ndarray')

        if len(pair_indices) == 0:
            return pairs

        for idx in range(len(pair_indices)):
            i, j = pair_indices[idx]
            pi = positions[i]
            pj = positions[j]
            spatial = np.linalg.norm(pi - pj)
            if spatial < min_spatial:
                continue
            fd = np.linalg.norm(features[i] - features[j])
            pairs.append((pi.copy(), pj.copy(), float(fd)))

        return pairs

    def _ransac_verify(self, pairs):
        if len(pairs) < self.RANSAC_MIN_INLIERS:
            return []

        pts_src = np.array([p[0] for p in pairs], dtype=np.float32)
        pts_dst = np.array([p[1] for p in pairs], dtype=np.float32)

        if len(pts_src) < 4:
            return pairs
        _, inlier_mask = cv2.estimateAffinePartial2D(
            pts_src.reshape(-1, 1, 2),
            pts_dst.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.RANSAC_INLIER_THRESHOLD,
            maxIters=2000,
            confidence=0.99,
        )

        if inlier_mask is None:
            return []

        inlier_mask = inlier_mask.ravel().astype(bool)
        verified = [pairs[i] for i in range(len(pairs)) if inlier_mask[i]]

        if len(verified) < self.RANSAC_MIN_INLIERS:
            return []

        return verified

    def _cluster_displacements(self, vectors):
        if len(vectors) < 12:
            return np.full((len(vectors),), -1, dtype=np.int32)
        eps = max(6.0, np.percentile(np.linalg.norm(vectors, axis=1), 20) * 0.35)
        labels = np.full((len(vectors),), -1, dtype=np.int32)
        cluster_id = 0
        for i in range(len(vectors)):
            if labels[i] != -1:
                continue
            distances = np.linalg.norm(vectors - vectors[i], axis=1)
            members = np.where(distances <= eps)[0]
            if len(members) >= 12:
                labels[members] = cluster_id
                cluster_id += 1
        return labels

    def _build_regions(self, pairs, labels, width, height):
        regions = []
        unique = [lab for lab in np.unique(labels) if lab >= 0]
        for lab in unique:
            idx = np.where(labels == lab)[0]
            if len(idx) < 12:
                continue

            pts1 = np.array([pairs[i][0] for i in idx], dtype=np.float32)
            pts2 = np.array([pairs[i][1] for i in idx], dtype=np.float32)
            all_pts = np.vstack([pts1, pts2])

            x_min = int(np.clip(np.min(all_pts[:, 0]), 0, width - 1))
            y_min = int(np.clip(np.min(all_pts[:, 1]), 0, height - 1))
            x_max = int(np.clip(np.max(all_pts[:, 0]), 0, width - 1))
            y_max = int(np.clip(np.max(all_pts[:, 1]), 0, height - 1))

            box_area = max(1, (x_max - x_min + 1) * (y_max - y_min + 1))
            confidence = min(100.0, (len(idx) * 4.0) + min(20.0, 25000.0 / box_area))

            regions.append(
                {
                    "x1": x_min / width,
                    "y1": y_min / height,
                    "x2": x_max / width,
                    "y2": y_max / height,
                    "confidence": round(float(confidence), 2),
                    "area": int(box_area),
                    "support_matches": int(len(idx)),
                }
            )

        regions.sort(key=lambda r: r["confidence"], reverse=True)
        return regions


class NoiseInconsistencyDetector:

    PATCH_SIZES = [16, 32, 64]

    def analyze(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to load image for noise analysis")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(gray_u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
        residual = gray - denoised.astype(np.float32)

        try:
            wavelet_sigma = estimate_sigma(gray_u8, channel_axis=None)
        except Exception:
            wavelet_sigma = float(np.std(residual))

        combined_z_map = None
        z_map_shape = None

        for patch in self.PATCH_SIZES:
            if h < patch or w < patch:
                continue

            stride = max(8, patch // 2)
            ny = (h - patch) // stride + 1
            nx = (w - patch) // stride + 1

            if ny < 2 or nx < 2:
                continue

            local_var_map = np.zeros((ny, nx), dtype=np.float32)
            for yi in range(ny):
                y = yi * stride
                for xi in range(nx):
                    x = xi * stride
                    block = residual[y : y + patch, x : x + patch]
                    local_var_map[yi, xi] = np.var(block)

            med = float(np.median(local_var_map))
            mad = float(np.median(np.abs(local_var_map - med))) + 1e-6
            z_map = 0.6745 * (local_var_map - med) / mad

            if z_map_shape is None:
                z_map_shape = z_map.shape
                combined_z_map = np.abs(z_map)
            else:
                resized = cv2.resize(np.abs(z_map), (z_map_shape[1], z_map_shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                combined_z_map = np.maximum(combined_z_map, resized)

        if combined_z_map is None:
            return {"score": 0.0, "regions": [], "residual_std": float(np.std(residual))}

        z_map_final = combined_z_map

        cross_channel_score = self._analyze_cross_channel_noise(image)

        kurtosis_score = self._analyze_kurtosis(residual)

        texture_level = float(np.std(z_map_final)) / (float(np.mean(z_map_final)) + 1e-6)
        z_cut = 3.0 if texture_level < 2.5 else 3.6

        suspicious_mask = (z_map_final > z_cut).astype(np.uint8) * 255
        suspicious_mask = cv2.morphologyEx(
            suspicious_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        suspicious_mask = cv2.morphologyEx(
            suspicious_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        contours, _ = cv2.findContours(suspicious_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        min_contour_area = max(10.0, (w * h) / 1_500_000.0)

        primary_patch = 32
        primary_stride = 16

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            px = x * primary_stride
            py = y * primary_stride
            pw = min(w - px, bw * primary_stride + primary_patch)
            ph = min(h - py, bh * primary_stride + primary_patch)
            if pw <= 0 or ph <= 0:
                continue

            region_z = z_map_final[y : y + bh, x : x + bw]
            confidence = min(100.0, float(np.mean(region_z) * 16.0))

            regions.append(
                {
                    "x1": px / w,
                    "y1": py / h,
                    "x2": (px + pw) / w,
                    "y2": (py + ph) / h,
                    "confidence": round(confidence, 2),
                    "area": int(pw * ph),
                }
            )

        regions.sort(key=lambda r: r["confidence"], reverse=True)

        suspicious_ratio = float(np.mean(z_map_final > z_cut))

        base_score = suspicious_ratio * 85.0
        region_score = min(12.0, len(regions) * 2.0)
        channel_score = cross_channel_score * 12.0
        kurt_score = kurtosis_score * 8.0

        score = min(100.0, base_score + region_score + channel_score + kurt_score)

        return {
            "score": round(float(score), 2),
            "regions": regions[:10],
            "residual_std": float(np.std(residual)),
            "suspicious_ratio": suspicious_ratio,
            "wavelet_sigma": float(wavelet_sigma),
            "cross_channel_inconsistency": float(cross_channel_score),
            "kurtosis_anomaly": float(kurtosis_score),
        }

    def _analyze_cross_channel_noise(self, image):
        try:
            b, g, r = cv2.split(image)
            channels = [r, g, b]

            sigmas = []
            for ch in channels:
                ch_u8 = ch.astype(np.uint8) if ch.dtype != np.uint8 else ch
                try:
                    sigma = estimate_sigma(ch_u8, channel_axis=None)
                except Exception:
                    sigma = float(np.std(ch.astype(np.float32)))
                sigmas.append(sigma)

            sigmas = np.array(sigmas, dtype=np.float64)

            if np.mean(sigmas) < 1e-6:
                return 0.0

            cv_channels = float(np.std(sigmas) / (np.mean(sigmas) + 1e-6))

            inconsistency = min(1.0, cv_channels / 0.7)
            return inconsistency

        except Exception:
            return 0.0

    def _analyze_kurtosis(self, residual, patch_size=32):
        try:
            h, w = residual.shape
            stride = patch_size
            if h < patch_size or w < patch_size:
                return 0.0

            kurtosis_values = []
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    block = residual[y : y + patch_size, x : x + patch_size].ravel()
                    k = scipy_kurtosis(block, fisher=True)
                    if np.isfinite(k):
                        kurtosis_values.append(k)

            if len(kurtosis_values) < 4:
                return 0.0

            kurtosis_values = np.array(kurtosis_values)

            med_k = np.median(kurtosis_values)
            mad_k = np.median(np.abs(kurtosis_values - med_k)) + 1e-6
            z_kurt = np.abs(0.6745 * (kurtosis_values - med_k) / mad_k)

            outlier_ratio = float(np.mean(z_kurt > 3.5))

            return min(1.0, outlier_ratio * 3.0)

        except Exception:
            return 0.0


def run_advanced_analyses(image_path):
    copy_move_detector = CopyMoveDetector()
    noise_detector = NoiseInconsistencyDetector()

    copy_move = copy_move_detector.analyze(image_path)
    noise = noise_detector.analyze(image_path)

    return {
        "copy_move": copy_move,
        "noise": noise,
    }
