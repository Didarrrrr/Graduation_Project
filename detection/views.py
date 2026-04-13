import json
from django.core.paginator import Paginator
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.urls import reverse
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
    upload_form = ImageUploadForm()
    settings_form = AnalysisSettingsForm()

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
    context = {
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
    return render(request, "detection/how_it_works.html")


@require_http_methods(["POST"])
def upload_image_view(request):
    form = ImageUploadForm(request.POST, request.FILES)

    if form.is_valid():
        uploaded_image = form.save()

        ela_quality = ELA_QUALITY
        sensitivity = DETECTION_SENSITIVITY
        include_metadata = True
        generate_heatmap = True
        settings_form = AnalysisSettingsForm(request.POST)
        if settings_form.is_valid():
            include_metadata = settings_form.cleaned_data["include_metadata"]
            generate_heatmap = settings_form.cleaned_data["generate_heatmap"]

        request.session["analysis_settings"] = {
            "ela_quality": ela_quality,
            "sensitivity": sensitivity,
            "include_metadata": include_metadata,
            "generate_heatmap": generate_heatmap,
        }

        return redirect("analyze_image", image_id=uploaded_image.id)
    else:
        for error in form.errors.values():
            messages.error(request, error)

        return redirect("home")


def analyze_image_view(request, image_id):
    uploaded_image = get_object_or_404(UploadedImage, id=image_id)

    analysis_settings = request.session.get(
        "analysis_settings",
        {
            "ela_quality": ELA_QUALITY,
            "sensitivity": DETECTION_SENSITIVITY,
            "include_metadata": True,
            "generate_heatmap": True,
        },
    )

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
    result = get_object_or_404(
        AnalysisResult.objects.select_related("image"),
        id=result_id,
    )

    suspicious_regions = result.suspicious_regions.all()[:5]

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
    all_analyses = AnalysisResult.objects.select_related("image").filter(
        analysis_completed_at__isnull=False
    ).order_by("-analysis_completed_at")

    total_count = all_analyses.count()
    forged_count = all_analyses.filter(forgery_status="forged").count()
    suspicious_count = all_analyses.filter(forgery_status="suspicious").count()
    authentic_count = all_analyses.filter(forgery_status="authentic").count()

    status_filter = request.GET.get("status")
    analyses = all_analyses
    if status_filter:
        analyses = analyses.filter(forgery_status=status_filter)

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


def download_report_view(request, result_id):
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
    result = get_object_or_404(AnalysisResult, id=result_id)

    image = result.image

    if result.ela_image:
        result.ela_image.delete(save=False)
    if result.heatmap_image:
        result.heatmap_image.delete(save=False)

    if image.original_image:
        image.original_image.delete(save=False)

    result.delete()
    image.delete()

    return redirect("gallery")
