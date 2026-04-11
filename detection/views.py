"""
Views for Image Forgery Detection System

Handles HTTP requests for:
- Image upload
- Analysis processing
- Results display
- About and documentation pages
"""

import json
from django.core.paginator import Paginator
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings
from .models import UploadedImage, AnalysisResult, MetadataAnalysis
from .pdf_report import build_analysis_report_pdf
from .forms import ImageUploadForm, AnalysisSettingsForm
from .pipeline import execute_full_analysis

ELA_QUALITY = getattr(settings, "ELA_QUALITY", 95)
DETECTION_SENSITIVITY = getattr(settings, "DETECTION_SENSITIVITY", 0.75)


def home_view(request):
    """
    Home page view - displays upload form and recent analyses.
    """
    upload_form = ImageUploadForm()
    settings_form = AnalysisSettingsForm()

    # Get recent analyses
    recent_analyses = AnalysisResult.objects.select_related("image").filter(
        analysis_completed_at__isnull=False
    ).order_by("-analysis_completed_at")[:6]

    context = {
        "upload_form": upload_form,
        "settings_form": settings_form,
        "recent_analyses": recent_analyses,
    }

    return render(request, "detection/home.html", context)


def about_view(request):
    """
    About page view - displays project information.
    """
    context = {
        "project_code": "GP_IT_16",
        "university": "Sulaimani Polytechnic University",
        "college": "Technical College of Informatics",
        "department": "Information Technology Department",
        "academic_year": "2025-2026",
        "supervisor": "Aras Masood Ismael",
        "team_members": [
            "Sahand Salih Raouf",
            "Didar Ibrahim Rasull",
            "Ayad Latif Hama",
        ],
    }

    return render(request, "detection/about.html", context)


def how_it_works_view(request):
    """
    How It Works page - explains the detection methodology.
    """
    return render(request, "detection/how_it_works.html")


@require_http_methods(["POST"])
def upload_image_view(request):
    """
    Handle image upload and initiate analysis.
    """
    form = ImageUploadForm(request.POST, request.FILES)

    if form.is_valid():
        # Save the uploaded image
        uploaded_image = form.save()

        # Use fixed analysis settings (frontend input removed intentionally).
        ela_quality = ELA_QUALITY
        sensitivity = DETECTION_SENSITIVITY
        include_metadata = True
        generate_heatmap = True
        settings_form = AnalysisSettingsForm(request.POST)
        if settings_form.is_valid():
            include_metadata = settings_form.cleaned_data["include_metadata"]
            generate_heatmap = settings_form.cleaned_data["generate_heatmap"]

        # Store settings in session
        request.session["analysis_settings"] = {
            "ela_quality": ela_quality,
            "sensitivity": sensitivity,
            "include_metadata": include_metadata,
            "generate_heatmap": generate_heatmap,
        }

        # Redirect to analysis page
        return redirect("analyze_image", image_id=uploaded_image.id)
    else:
        # Display form errors
        for error in form.errors.values():
            messages.error(request, error)

        return redirect("home")


def analyze_image_view(request, image_id):
    """
    Perform forgery analysis on uploaded image.
    """
    uploaded_image = get_object_or_404(UploadedImage, id=image_id)

    # Get analysis settings from session
    analysis_settings = request.session.get(
        "analysis_settings",
        {
            "ela_quality": ELA_QUALITY,
            "sensitivity": DETECTION_SENSITIVITY,
            "include_metadata": True,
            "generate_heatmap": True,
        },
    )

    # Check if analysis already exists
    existing_result = AnalysisResult.objects.filter(image=uploaded_image).first()
    if existing_result and existing_result.analysis_completed_at:
        return redirect("analysis_result", result_id=existing_result.id)

    try:
        result = execute_full_analysis(uploaded_image, analysis_settings)
        return redirect("analysis_result", result_id=result.id)

    except Exception as e:
        messages.error(request, f"Analysis failed: {str(e)}")
        return redirect("home")


def analysis_result_view(request, result_id):
    """
    Display analysis results.
    """
    result = get_object_or_404(
        AnalysisResult.objects.select_related("image"),
        id=result_id,
    )

    suspicious_regions = result.suspicious_regions.all()[:5]

    # Result notes store per-detector scores (JSON) created during analysis.
    component_scores = {}
    if result.notes:
        try:
            component_scores = json.loads(result.notes)
        except Exception:
            component_scores = {}

    try:
        metadata = result.metadata_analysis
    except MetadataAnalysis.DoesNotExist:
        metadata = None

    context = {
        "result": result,
        "image": result.image,
        "suspicious_regions": suspicious_regions,
        "metadata": metadata,
        "component_scores": component_scores,
    }

    return render(request, "detection/result.html", context)


def gallery_view(request):
    """
    Display gallery of analyzed images.
    """
    all_analyses = AnalysisResult.objects.select_related("image").filter(
        analysis_completed_at__isnull=False
    ).order_by("-analysis_completed_at")

    # Compute stats from unfiltered queryset so totals are always accurate
    total_count = all_analyses.count()
    forged_count = all_analyses.filter(forgery_status="forged").count()
    suspicious_count = all_analyses.filter(forgery_status="suspicious").count()
    authentic_count = all_analyses.filter(forgery_status="authentic").count()

    # Filter by status if requested
    status_filter = request.GET.get("status")
    analyses = all_analyses
    if status_filter:
        analyses = analyses.filter(forgery_status=status_filter)

    # Pagination (12 items per page)
    paginator = Paginator(analyses, 12)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        "analyses": page_obj,
        "page_obj": page_obj,
        "status_filter": status_filter,
        "total_count": total_count,
        "forged_count": forged_count,
        "suspicious_count": suspicious_count,
        "authentic_count": authentic_count,
    }

    return render(request, "detection/gallery.html", context)

@csrf_exempt
def api_analyze_view(request):
    """
    API endpoint for image analysis (AJAX).
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    if "image" not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    try:
        # Save uploaded image
        uploaded_image = UploadedImage.objects.create(
            original_image=request.FILES["image"], filename=request.FILES["image"].name
        )

        api_settings = {
            "ela_quality": ELA_QUALITY,
            "sensitivity": DETECTION_SENSITIVITY,
            "include_metadata": True,
            "generate_heatmap": True,
        }
        result = execute_full_analysis(uploaded_image, api_settings)
        notes = {}
        if result.notes:
            try:
                notes = json.loads(result.notes)
            except Exception:
                pass

        combined_score = result.confidence_score or 0.0
        # Same label as the web UI and DB (pipeline sets forgery_status).
        status = result.forgery_status
        if status == "pending":
            status = "authentic"

        meta = MetadataAnalysis.objects.filter(analysis_result=result).first()
        response = {
            "status": "success",
            "image_id": uploaded_image.id,
            "result_id": result.id,
            "forgery_status": status,
            "confidence_score": round(float(combined_score), 2),
            "ela_score": round(float(result.ela_score or 0), 2),
            "metadata_score": round(float(meta.metadata_score), 2)
            if meta
            else None,
            "copy_move_score": float(notes.get("copy_move_score", 0)),
            "noise_score": float(notes.get("noise_score", 0)),
            "copy_move_match_count": notes.get("copy_move_matches", 0),
            "noise_suspicious_ratio": round(
                float(notes.get("noise_suspicious_ratio", 0.0)), 4
            ),
            "fusion": notes.get("fusion"),
            "suspicious_regions_count": result.suspicious_regions.count(),
            "editing_software_detected": meta.editing_software_found if meta else False,
            "software_detected": meta.software_detected if meta else "",
            "timestamp_inconsistent": meta.timestamp_inconsistent if meta else False,
            "result_url": reverse("analysis_result", args=[result.id]),
        }

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def download_report_view(request, result_id):
    """
    Generate and download analysis report as a PDF file.
    """
    result = get_object_or_404(AnalysisResult, id=result_id)
    component_scores = {}
    if result.notes:
        try:
            component_scores = json.loads(result.notes)
        except Exception:
            component_scores = {}

    buffer = build_analysis_report_pdf(result, component_scores)
    response = HttpResponse(buffer, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="forgery_report_{result_id}.pdf"'
    )
    return response


@require_http_methods(["POST"])
def delete_analysis_view(request, result_id):
    """
    Delete an analysis result and associated images.
    """
    result = get_object_or_404(AnalysisResult, id=result_id)

    image = result.image

    # Delete associated files
    if result.ela_image:
        result.ela_image.delete(save=False)
    if result.heatmap_image:
        result.heatmap_image.delete(save=False)

    # Delete original image
    if image.original_image:
        image.original_image.delete(save=False)

    # Delete database records
    result.delete()
    image.delete()

    return redirect("gallery")
