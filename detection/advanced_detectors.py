"""
Advanced forgery detectors:
- Copy-move detection with block-based DCT features, KDTree matching, and RANSAC verification
- PRNU-inspired noise inconsistency analysis with wavelet estimation and cross-channel analysis

Improvements:
  Copy-Move:
    - 16 zigzag DCT coefficients (from 8) for better block identity
    - scipy KDTree for efficient radius-based matching
    - RANSAC geometric verification to filter false positives
    - Finer quantization (1/100 step) preserving detail
    - Otsu's adaptive variance filtering for flat-block removal
    - Multi-scale blocks (8 and 16) for detecting clones at different sizes
    
  Noise:
    - Wavelet-based noise estimation (Donoho's MAD estimator)
    - Multi-scale analysis (16, 32, 64 patches)
    - Cross-channel (R, G, B) noise analysis
    - Kurtosis-based detection for non-Gaussian residuals
    - Non-local means denoising for better residual extraction
    - Calibrated scoring combining multiple signals
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kurtosis as scipy_kurtosis
from skimage.restoration import estimate_sigma


# ────────────────────────────────────────────────────────────────────────────
# Zigzag order for an NxN DCT block (used to pick the first K AC coefficients)
# ────────────────────────────────────────────────────────────────────────────

def _zigzag_indices(n, k):
    """Return the first *k* (row, col) indices in zigzag order, skipping DC (0,0)."""
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
    # Skip DC (0,0), take next k
    ac_indices = [(r, c) for r, c in indices if not (r == 0 and c == 0)]
    return ac_indices[:k]


class CopyMoveDetector:
    """Detect copy-move manipulations using block-based DCT matching with geometric verification."""

    # Number of AC DCT coefficients per block (zigzag order)
    N_COEFFS = 16
    # Quantization step (finer = more discriminative)
    QUANT_STEP = 100.0
    # KDTree radius for matching — tightened from 0.10 to reduce texture false matches
    MATCH_RADIUS = 0.07
    # Minimum spatial distance between matched blocks — raised to 3x to avoid
    # neighbouring blocks that naturally look alike in uniform areas
    MIN_SPATIAL_DIST_FACTOR = 3.0
    # RANSAC inlier threshold (pixels) — tightened from 6.0 to demand closer geometric fit
    RANSAC_INLIER_THRESHOLD = 4.0
    # Minimum inliers to accept a RANSAC model — raised from 6 to 12 so
    # repetitive textures with scattered matches don't pass verification
    RANSAC_MIN_INLIERS = 12

    def __init__(self, block_size=16, step=8):
        self.block_size = block_size
        self.step = step
        # Precompute zigzag indices for max block size
        self._zigzag_16 = _zigzag_indices(16, self.N_COEFFS)
        self._zigzag_8 = _zigzag_indices(8, self.N_COEFFS)

    def analyze(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to load image for copy-move analysis")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # --- Multi-scale: run at block_size 8 and 16, merge results ---
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

            # Geometric verification (RANSAC)
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

        # Conservative calibration — require strong cluster support to score high.
        # dominant_cluster_size must be ≥ 15 to contribute meaningfully.
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

                # Extract N_COEFFS zigzag AC coefficients
                n_avail = min(len(zigzag_indices), self.N_COEFFS)
                signature = np.array(
                    [dct[r, c] for r, c in zigzag_indices[:n_avail]],
                    dtype=np.float32,
                )

                # Pad if needed (for very small blocks)
                if len(signature) < self.N_COEFFS:
                    signature = np.pad(signature, (0, self.N_COEFFS - len(signature)))

                nrm = float(np.linalg.norm(signature))
                if nrm > 1e-6:
                    signature = signature / nrm

                # Finer quantization (1/100 step)
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

        # Adaptive variance filtering using Otsu's method
        variances = np.array([b["variance"] for b in raw_blocks], dtype=np.float64)
        var_threshold = self._otsu_variance_threshold(variances)

        blocks = [b for b in raw_blocks if b["variance"] >= var_threshold]
        return blocks

    def _otsu_variance_threshold(self, variances):
        """
        Use Otsu's method to separate flat/textured blocks instead of fixed percentile.
        This adaptively finds the optimal threshold between low-variance (flat) and
        high-variance (textured) blocks.
        """
        if len(variances) < 10:
            return max(0.0015, float(np.percentile(variances, 35)))

        # Normalize variances to 0-255 for Otsu
        v_min, v_max = np.min(variances), np.max(variances)
        if v_max - v_min < 1e-8:
            return float(v_min)

        normalized = ((variances - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Map back to original scale
        threshold = v_min + (otsu_val / 255.0) * (v_max - v_min)

        # Don't let Otsu go too low (noise floor)
        return max(0.001, float(threshold))

    def _match_kdtree(self, features, positions, block_size):
        """
        Use scipy KDTree for efficient radius-based matching.
        Replaces lexicographic sort + linear scan — finds ALL neighbors within
        the distance threshold, not just adjacent ones in sort order.
        """
        tree = cKDTree(features)
        pairs = []
        min_spatial = block_size * self.MIN_SPATIAL_DIST_FACTOR

        # Query pairs within radius
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
        """
        RANSAC-based geometric verification.
        True copy-move regions share a consistent affine transform.
        Filters out false matches from repetitive textures.
        """
        if len(pairs) < self.RANSAC_MIN_INLIERS:
            return []  # Too few matches to verify — reject rather than pass through

        pts_src = np.array([p[0] for p in pairs], dtype=np.float32)
        pts_dst = np.array([p[1] for p in pairs], dtype=np.float32)

        if len(pts_src) < 4:
            return pairs

        # Use estimateAffinePartial2D for robust RANSAC estimation
        # (handles translation + rotation + scale = copy-move transforms)
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

        # Only accept if we have enough inliers
        if len(verified) < self.RANSAC_MIN_INLIERS:
            return []

        return verified

    def _cluster_displacements(self, vectors):
        # Raised cluster minimum from 8 → 12 to reject small spurious groups
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
    """
    PRNU-inspired residual noise inconsistency detector.
    
    Improvements:
    - Wavelet-based noise estimation (scikit-image)
    - Multi-scale patch analysis (16, 32, 64)
    - Cross-channel (R, G, B) noise correlation
    - Kurtosis-based anomaly detection
    - Non-local means denoising for better residual extraction
    """

    PATCH_SIZES = [16, 32, 64]

    def analyze(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to load image for noise analysis")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- Non-local means denoising for better residual extraction ---
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(gray_u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
        residual = gray - denoised.astype(np.float32)

        # --- Wavelet-based noise estimation (Donoho's MAD estimator) ---
        try:
            # estimate_sigma works on the image directly
            wavelet_sigma = estimate_sigma(gray_u8, channel_axis=None)
        except Exception:
            wavelet_sigma = float(np.std(residual))

        # --- Multi-scale analysis ---
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

            # Resize z_map to a common grid for combining
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

        # --- Cross-channel noise analysis ---
        cross_channel_score = self._analyze_cross_channel_noise(image)

        # --- Kurtosis-based detection ---
        kurtosis_score = self._analyze_kurtosis(residual)

        # Adaptive z threshold based on texture
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

        # Use the primary patch/stride for coordinate mapping
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

        # --- Improved scoring: reduced multipliers to cut false positives ---
        # Base score from suspicious pixel ratio (reduced from 120)
        base_score = suspicious_ratio * 85.0
        # Region contribution (reduced from 3.0 per region)
        region_score = min(12.0, len(regions) * 2.0)
        # Cross-channel inconsistency (reduced from 25; cameras naturally differ slightly)
        channel_score = cross_channel_score * 12.0
        # Kurtosis outlier score (reduced from 20; natural textures can be non-Gaussian)
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
        """
        Estimate noise independently in R, G, B channels.
        Authentic images show correlated noise across channels;
        spliced regions break this correlation.
        
        Returns:
            float: 0-1 inconsistency score (higher = more suspicious)
        """
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

            # Coefficient of variation of channel noise levels
            # Authentic images: channels have very similar noise => low CV
            # Spliced images: pasted region may have different channel noise ratios
            cv_channels = float(np.std(sigmas) / (np.mean(sigmas) + 1e-6))

            # Normalize to 0-1 range — widened from /0.5 to /0.7 because
            # authentic sensor channels naturally differ by 10-20% in noise level
            inconsistency = min(1.0, cv_channels / 0.7)
            return inconsistency

        except Exception:
            return 0.0

    def _analyze_kurtosis(self, residual, patch_size=32):
        """
        Analyze block-level kurtosis of noise residuals.
        Natural image noise follows approximately Gaussian distribution (kurtosis ≈ 0 for excess).
        Manipulated regions often show different kurtosis.
        
        Returns:
            float: 0-1 anomaly score (higher = more suspicious)
        """
        try:
            h, w = residual.shape
            stride = patch_size
            if h < patch_size or w < patch_size:
                return 0.0

            kurtosis_values = []
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    block = residual[y : y + patch_size, x : x + patch_size].ravel()
                    k = scipy_kurtosis(block, fisher=True)  # excess kurtosis (0 for Gaussian)
                    if np.isfinite(k):
                        kurtosis_values.append(k)

            if len(kurtosis_values) < 4:
                return 0.0

            kurtosis_values = np.array(kurtosis_values)

            # Count outlier blocks (excess kurtosis far from 0)
            med_k = np.median(kurtosis_values)
            mad_k = np.median(np.abs(kurtosis_values - med_k)) + 1e-6
            z_kurt = np.abs(0.6745 * (kurtosis_values - med_k) / mad_k)

            outlier_ratio = float(np.mean(z_kurt > 3.5))

            # Normalize to 0-1 (reduced multiplier from 5 → 3 to cut texture false positives)
            return min(1.0, outlier_ratio * 3.0)

        except Exception:
            return 0.0


def run_advanced_analyses(image_path):
    """Run copy-move + noise analysis and return unified payload."""
    copy_move_detector = CopyMoveDetector()
    noise_detector = NoiseInconsistencyDetector()

    copy_move = copy_move_detector.analyze(image_path)
    noise = noise_detector.analyze(image_path)

    return {
        "copy_move": copy_move,
        "noise": noise,
    }
