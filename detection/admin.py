from django.contrib import admin
from .models import UploadedImage, AnalysisResult, MetadataAnalysis, SuspiciousRegion
from django.utils.html import format_html

@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'filename', 'image_format', 'file_size_formatted', 'uploaded_at', 'processed']
    list_filter = ['processed', 'image_format', 'uploaded_at']
    search_fields = ['filename', 'id']
    readonly_fields = ['uploaded_at', 'file_size', 'width', 'height']
    date_hierarchy = 'uploaded_at'
    
    def file_size_formatted(self, obj):
        if obj.file_size:
            if obj.file_size < 1024:
                return f"{obj.file_size} B"
            elif obj.file_size < 1024 * 1024:
                return f"{obj.file_size / 1024:.2f} KB"
            else:
                return f"{obj.file_size / (1024 * 1024):.2f} MB"
        return "Unknown"
    file_size_formatted.short_description = 'File Size'


class SuspiciousRegionInline(admin.TabularInline):
    model = SuspiciousRegion
    extra = 0
    readonly_fields = ['confidence', 'area_size', 'detection_method']
    fields = ['x1', 'y1', 'x2', 'y2', 'confidence', 'detection_method']


class MetadataAnalysisInline(admin.StackedInline):
    model = MetadataAnalysis
    extra = 0
    readonly_fields = [
        'software_detected', 'editing_software_found',
        'camera_make', 'camera_model',
        'datetime_original', 'datetime_digitized', 'datetime_modified',
        'timestamp_inconsistent',
        'metadata_score', 'suspicious_indicators'
    ]
    fieldsets = (
        ('Software Information', {
            'fields': ('software_detected', 'editing_software_found')
        }),
        ('Camera Information', {
            'fields': ('camera_make', 'camera_model')
        }),
        ('Timestamp Information', {
            'fields': ('datetime_original', 'datetime_digitized', 'datetime_modified', 'timestamp_inconsistent')
        }),
        ('Analysis Results', {
            'fields': ('metadata_score', 'suspicious_indicators')
        }),
    )


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'image_id', 'forgery_status_badge', 'confidence_score',
        'ela_score', 'processing_time_formatted', 'analysis_completed_at'
    ]
    list_filter = ['forgery_status', 'ela_threshold_exceeded', 'analysis_completed_at']
    search_fields = ['image__filename', 'id', 'notes']
    readonly_fields = [
        'analysis_started_at', 'analysis_completed_at',
        'processing_time', 'ela_score', 'confidence_score'
    ]
    inlines = [MetadataAnalysisInline, SuspiciousRegionInline]
    date_hierarchy = 'analysis_completed_at'
    
    fieldsets = (
        ('Image Information', {
            'fields': ('image',)
        }),
        ('Analysis Status', {
            'fields': ('forgery_status', 'confidence_score', 'notes')
        }),
        ('ELA Analysis', {
            'fields': ('ela_score', 'ela_threshold_exceeded', 'ela_image')
        }),
        ('Timing Information', {
            'fields': ('analysis_started_at', 'analysis_completed_at', 'processing_time')
        }),
    )
    
    def image_id(self, obj):
        return f"Image #{obj.image.id}"
    image_id.short_description = 'Image'
    
    def forgery_status_badge(self, obj):
        colors = {
            'authentic': 'green',
            'suspicious': 'orange',
            'forged': 'red',
            'pending': 'gray'
        }
        color = colors.get(obj.forgery_status, 'black')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color, obj.get_forgery_status_display()
        )
    forgery_status_badge.short_description = 'Status'
    
    def processing_time_formatted(self, obj):
        if obj.processing_time:
            return f"{obj.processing_time:.2f}s"
        return "N/A"
    processing_time_formatted.short_description = 'Processing Time'


@admin.register(MetadataAnalysis)
class MetadataAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'analysis_result_id', 'editing_software_found',
        'software_detected_short', 'metadata_score', 'timestamp_inconsistent'
    ]
    list_filter = ['editing_software_found', 'timestamp_inconsistent']
    search_fields = ['software_detected', 'camera_make', 'camera_model']
    readonly_fields = ['metadata_json']
    
    def analysis_result_id(self, obj):
        return f"Analysis #{obj.analysis_result.id}"
    analysis_result_id.short_description = 'Analysis'
    
    def software_detected_short(self, obj):
        if obj.software_detected:
            return obj.software_detected[:30] + '...' if len(obj.software_detected) > 30 else obj.software_detected
        return "None"
    software_detected_short.short_description = 'Software'


@admin.register(SuspiciousRegion)
class SuspiciousRegionAdmin(admin.ModelAdmin):
    list_display = ['id', 'analysis_result_id', 'confidence', 'area_size', 'detection_method']
    list_filter = ['detection_method', 'confidence']
    search_fields = ['analysis_result__image__filename']
    
    def analysis_result_id(self, obj):
        return f"Analysis #{obj.analysis_result.id}"
    analysis_result_id.short_description = 'Analysis'