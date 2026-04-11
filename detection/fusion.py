"""
Fuse the four forensic signals (ELA, metadata, copy-move DCT, noise) into one score.

Improved fusion with:
- Asymmetric metadata trust: positive findings (software detected, Q-table issues)
  are trusted at near-face-value; only "clean" verdicts are dampened by EXIF completeness
- Metadata-pixel synergy: when editing software is confirmed AND pixel detectors
  show above-baseline signals, apply a direct synergy boost
- Strong-signal boost: when any detector is confident (>60) and corroborated, lift
- Provenance-aware dampening: when metadata confirms camera pipeline, suppress
  pixel-level false positives
- Soft agreement scoring (continuous corroboration metric)
- Consensus clean ceiling for authentic images
"""


def fuse_four_method_scores(
    ela_score,
    metadata_score,
    copy_move_score,
    noise_score,
    *,
    metadata_confidence=0.0,
    copy_match_count=0,
    noise_suspicious_ratio=0.0,
    noise_cross_channel=0.0,
    include_metadata=True,
):
    """
    Combine sub-scores using adaptive weights that respond to signal reliability,
    with asymmetric metadata trust, metadata-pixel synergy, provenance-aware
    dampening, and strong-signal lift.

    Returns:
        (combined_score: float, diagnostics: dict)
    """
    ela = float(ela_score or 0.0)
    meta_raw = float(metadata_score or 0.0)
    copy_raw = float(copy_move_score or 0.0)
    noise_raw = float(noise_score or 0.0)

    mc = max(0.0, min(1.0, float(metadata_confidence or 0.0)))

    # ─── Asymmetric metadata trust ────────────────────────────────────────
    # When metadata finds positive evidence of manipulation (software, timestamps,
    # Q-table issues → score > 45), trust it near-fully — the finding itself is
    # reliable regardless of how many other EXIF tags exist.
    # When metadata says "clean" (low score), dampen by EXIF completeness.
    if meta_raw >= 45.0:
        # Positive finding: trust it (floor at 0.88)
        meta_trust = max(0.88, 0.30 + 0.70 * mc)
        meta_adj = meta_raw * meta_trust
    else:
        meta_trust = 0.30 + 0.70 * mc
        meta_adj = meta_raw * meta_trust

    matches = int(copy_match_count or 0)
    copy_trust = min(1.0, matches / 18.0) if matches < 18 else 1.0
    copy_adj = copy_raw * (0.50 + 0.50 * copy_trust)

    nsr = max(0.0, min(1.0, float(noise_suspicious_ratio or 0.0)))
    noise_damp = 1.0 - 0.28 * max(0.0, min(1.0, (nsr - 0.22) / 0.35))
    noise_adj = noise_raw * noise_damp

    ncc = max(0.0, min(1.0, float(noise_cross_channel or 0.0)))

    # ─── Provenance-aware dampening ───────────────────────────────────────
    provenance_dampening = 1.0
    strong_provenance = (
        include_metadata
        and mc >= 0.75
        and meta_raw < 35.0
    )
    if strong_provenance:
        provenance_dampening = 0.55 + 0.45 * (meta_raw / 35.0)
        ela = ela * provenance_dampening
        copy_adj = copy_adj * provenance_dampening
        noise_adj = noise_adj * provenance_dampening

    ela_reliability = min(1.0, ela / 80.0)

    # Compositing/splice mode detection
    compositing_splice_mode = (
        ela >= 54.0
        and copy_adj < 22.0
        and matches < 12
        and (
            noise_adj >= 20.0
            or ela >= 68.0
            or ncc >= 0.5
        )
    )

    if include_metadata:
        if compositing_splice_mode:
            w_ela, w_meta, w_copy, w_noise = 0.52, 0.20, 0.06, 0.22
        else:
            base_ela_w = 0.38 + 0.06 * ela_reliability
            remaining = 1.0 - base_ela_w
            w_ela = base_ela_w
            w_meta = remaining * 0.33
            w_copy = remaining * 0.34
            w_noise = remaining * 0.33
        combined = (
            ela * w_ela
            + meta_adj * w_meta
            + copy_adj * w_copy
            + noise_adj * w_noise
        )
        parts_for_agreement = [ela, meta_adj, copy_adj, noise_adj]
    else:
        if compositing_splice_mode:
            w_ela, w_copy, w_noise = 0.72, 0.08, 0.20
        else:
            base_ela_w = 0.58 + 0.06 * ela_reliability
            remaining = 1.0 - base_ela_w
            w_ela = base_ela_w
            w_copy = remaining * 0.50
            w_noise = remaining * 0.50
        combined = ela * w_ela + copy_adj * w_copy + noise_adj * w_noise
        parts_for_agreement = [ela, copy_adj, noise_adj]

    # --- Soft agreement scoring ---
    thr_high = 42.0
    thr_low = 20.0

    soft_high = sum(
        min(1.0, max(0.0, (x - thr_high) / 22.0))
        for x in parts_for_agreement
    )
    max_part = max(parts_for_agreement) if parts_for_agreement else 0.0
    low_count = sum(1 for x in parts_for_agreement if x < thr_low)
    high_count = sum(1 for x in parts_for_agreement if x >= thr_high)

    # Soft corroboration bonus
    if len(parts_for_agreement) >= 4:
        if soft_high >= 2.5:
            combined = min(100.0, combined + min(12.0, soft_high * 3.5))
        elif soft_high >= 1.5 and max_part >= 55.0:
            combined = min(100.0, combined + min(8.0, soft_high * 3.0))
        elif soft_high >= 0.8 and max_part >= 55.0:
            combined = min(100.0, combined + min(5.0, soft_high * 2.5))
    elif len(parts_for_agreement) >= 3:
        if soft_high >= 2.0:
            combined = min(100.0, combined + min(8.0, soft_high * 3.0))
        elif soft_high >= 1.0 and max_part >= 55.0:
            combined = min(100.0, combined + min(5.0, soft_high * 2.5))

    # Very high ELA with strong agreement
    if len(parts_for_agreement) >= 4 and high_count >= 3 and ela >= 80.0:
        combined = min(100.0, combined + 5.0)

    # ─── Strong-signal lift ───────────────────────────────────────────────
    # When any single detector is confident (>60) and at least one other
    # method corroborates (>38), give a direct boost.
    # Threshold lowered from 70 to 60 so metadata findings (~63 after trust)
    # can drive the lift when pixel detectors partially corroborate.
    # Authentic images never trigger this: their max_part is < 35.
    if max_part >= 60.0:
        supporting = sum(
            1 for x in parts_for_agreement
            if x >= 38.0 and x < max_part - 5.0
        )
        if supporting >= 2:
            boost = min(16.0, (max_part - 45.0) * 0.45)
            combined = min(100.0, combined + boost)
        elif supporting >= 1:
            boost = min(11.0, (max_part - 50.0) * 0.35)
            combined = min(100.0, combined + max(0.0, boost))

    # ─── Metadata-pixel synergy ───────────────────────────────────────────
    # When metadata strongly indicates editing (meta_adj >= 50, i.e. Photoshop,
    # GIMP, etc. confirmed) AND pixel-level detectors show above-baseline signals,
    # this is powerful evidence of manipulation.
    # An authentic Photoshop-exported photo has pixel scores < 25 (no manipulation).
    # A forged Photoshop image has pixel scores 30-55 (recompression dampens them
    # but doesn't eliminate them entirely).
    # This synergy ONLY fires when metadata explicitly found something (>= 50 after
    # trust) — it cannot fire for authentic images (meta_adj < 30 typically).
    if include_metadata and meta_adj >= 50.0:
        pixel_scores = [ela, copy_adj, noise_adj]
        above_baseline = sum(1 for x in pixel_scores if x >= 28.0)
        avg_pixel = sum(pixel_scores) / 3.0
        if above_baseline >= 2 and avg_pixel >= 30.0:
            synergy = min(20.0, (meta_adj - 40.0) * 0.45 + avg_pixel * 0.30)
            combined = min(100.0, combined + synergy)
        elif above_baseline >= 1 and avg_pixel >= 25.0:
            synergy = min(12.0, (meta_adj - 40.0) * 0.25 + avg_pixel * 0.20)
            combined = min(100.0, combined + synergy)

    # ─── Consensus "clean" ────────────────────────────────────────────────
    need_weak = 3 if len(parts_for_agreement) >= 4 else 2
    if low_count >= need_weak and max_part < 28.0:
        combined = min(combined, 35.0)

    # ─── Single-channel outlier blending ──────────────────────────────────
    if len(parts_for_agreement) >= 4 and not compositing_splice_mode:
        s = sorted(parts_for_agreement)
        median = 0.5 * (s[1] + s[2])
        if s[3] - s[0] > 50.0 and s[2] < 20.0:
            combined = 0.60 * combined + 0.40 * median

    combined = round(min(100.0, max(0.0, combined)), 2)

    diagnostics = {
        "metadata_adjusted": round(meta_adj, 2) if include_metadata else None,
        "copy_move_adjusted": round(copy_adj, 2),
        "noise_adjusted": round(noise_adj, 2),
        "agreement_high_methods": high_count,
        "soft_agreement": round(soft_high, 2),
        "compositing_splice_mode": compositing_splice_mode,
        "ela_reliability": round(ela_reliability, 2),
        "provenance_dampening": round(provenance_dampening, 3),
        "strong_provenance": strong_provenance,
    }
    return combined, diagnostics
