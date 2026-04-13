import json
import time

from django.core.files.base import ContentFile
from django.utils import timezone
from io import BytesIO

from .models import UploadedImage, AnalysisResult, MetadataAnalysis, SuspiciousRegion
from .ela_detector import perform_ela_analysis
from .metadata_analyzer import analyze_metadata
from .advanced_detectors import run_advanced_analyses
from .fusion import fuse_four_method_scores

SCORE_FORGED_MIN = 70.0
SCORE_SUSPICIOUS_MIN = 40.0


def execute_full_analysis(uploaded_image: UploadedImage, analysis_settings: dict) -> AnalysisResult:
    result, _ = AnalysisResult.objects.get_or_create(
        image=uploaded_image,
        defaults={
            "analysis_started_at": timezone.now(),
            "forgery_status": "pending",
        },
    )

    start_time = time.time()
    image_path = uploaded_image.original_image.path

    SuspiciousRegion.objects.filter(analysis_result=result).delete()
    MetadataAnalysis.objects.filter(analysis_result=result).delete()

    ela_results = perform_ela_analysis(
        image_path,
        quality=analysis_settings["ela_quality"],
        sensitivity=analysis_settings["sensitivity"],
    )

    ela_buffer = BytesIO()
    ela_results["ela_image"].save(ela_buffer, format="PNG")
    ela_buffer.seek(0)
    result.ela_image.save(
        f"ela_{uploaded_image.id}.png", ContentFile(ela_buffer.read())
    )

    if analysis_settings.get("generate_heatmap", True):
        heatmap_buffer = BytesIO()
        ela_results["heatmap"].save(heatmap_buffer, format="PNG")
        heatmap_buffer.seek(0)
        result.heatmap_image.save(
            f"heatmap_{uploaded_image.id}.png", ContentFile(heatmap_buffer.read())
        )
    else:
        if result.heatmap_image:
            result.heatmap_image.delete(save=False)
        result.heatmap_image = None

    result.ela_score = ela_results["score"]
    result.ela_threshold_exceeded = ela_results["threshold_exceeded"]

    for region in ela_results["suspicious_regions"][:10]:
        SuspiciousRegion.objects.create(
            analysis_result=result,
            x1=region["x1"],
            y1=region["y1"],
            x2=region["x2"],
            y2=region["y2"],
            confidence=region["confidence"],
            area_size=region.get("area"),
            detection_method="ELA",
        )

    advanced_results = run_advanced_analyses(image_path)
    copy_move_results = advanced_results.get("copy_move", {})
    noise_results = advanced_results.get("noise", {})

    for region in copy_move_results.get("regions", [])[:5]:
        SuspiciousRegion.objects.create(
            analysis_result=result,
            x1=region["x1"],
            y1=region["y1"],
            x2=region["x2"],
            y2=region["y2"],
            confidence=region["confidence"],
            area_size=region.get("area"),
            detection_method="COPY_MOVE",
        )

    for region in noise_results.get("regions", [])[:5]:
        SuspiciousRegion.objects.create(
            analysis_result=result,
            x1=region["x1"],
            y1=region["y1"],
            x2=region["x2"],
            y2=region["y2"],
            confidence=region["confidence"],
            area_size=region.get("area"),
            detection_method="NOISE",
        )

    ela_score = ela_results["score"]
    copy_move_score = copy_move_results.get("score", 0)
    noise_score = noise_results.get("score", 0)

    metadata_results = {}
    if analysis_settings.get("include_metadata", True):
        metadata_results = analyze_metadata(image_path)
        MetadataAnalysis.objects.create(
            analysis_result=result,
            software_detected=metadata_results.get("software_detected", ""),
            editing_software_found=metadata_results.get(
                "editing_software_found", False
            ),
            camera_make=metadata_results.get("camera_make", ""),
            camera_model=metadata_results.get("camera_model", ""),
            datetime_original=metadata_results.get("datetime_original", ""),
            datetime_digitized=metadata_results.get("datetime_digitized", ""),
            datetime_modified=metadata_results.get("datetime_modified", ""),
            timestamp_inconsistent=metadata_results.get(
                "timestamp_inconsistent", False
            ),
            metadata_json=metadata_results.get("metadata_json", "{}"),
            suspicious_indicators=metadata_results.get("suspicious_indicators", ""),
            metadata_score=metadata_results.get("metadata_score", 0),
        )

        metadata_score = metadata_results.get("metadata_score", 0)
        combined_score, fusion_diag = fuse_four_method_scores(
            ela_score,
            metadata_score,
            copy_move_score,
            noise_score,
            metadata_confidence=metadata_results.get("metadata_confidence", 0.0),
            copy_match_count=copy_move_results.get("match_count", 0),
            noise_suspicious_ratio=noise_results.get("suspicious_ratio", 0.0),
            noise_cross_channel=noise_results.get("cross_channel_inconsistency", 0.0),
            include_metadata=True,
        )
    else:
        combined_score, fusion_diag = fuse_four_method_scores(
            ela_score,
            0,
            copy_move_score,
            noise_score,
            copy_match_count=copy_move_results.get("match_count", 0),
            noise_suspicious_ratio=noise_results.get("suspicious_ratio", 0.0),
            noise_cross_channel=noise_results.get("cross_channel_inconsistency", 0.0),
            include_metadata=False,
        )

    if combined_score >= SCORE_FORGED_MIN:
        result.forgery_status = "forged"
    elif combined_score >= SCORE_SUSPICIOUS_MIN:
        result.forgery_status = "suspicious"
    else:
        result.forgery_status = "authentic"

    result.confidence_score = combined_score
    result.notes = json.dumps(
        {
            "fusion": fusion_diag,
            "ela_threshold_used": ela_results.get("ela_threshold_used"),
            "ela_score": round(float(ela_score), 2),
            "metadata_score": round(
                float(metadata_results.get("metadata_score", 0)), 2
            )
            if analysis_settings.get("include_metadata", True)
            else None,
            "metadata_confidence": round(
                float(metadata_results.get("metadata_confidence", 0.0)) * 100.0,
                2,
            )
            if analysis_settings.get("include_metadata", True)
            else None,
            "copy_move_score": round(float(copy_move_score), 2),
            "noise_score": round(float(noise_score), 2),
            "copy_move_matches": copy_move_results.get("match_count", 0),
            "noise_suspicious_ratio": noise_results.get("suspicious_ratio", 0.0),
        }
    )
    result.analysis_completed_at = timezone.now()
    result.processing_time = time.time() - start_time
    result.save()

    uploaded_image.processed = True
    uploaded_image.save(update_fields=["processed"])

    return result
