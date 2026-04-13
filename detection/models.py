from django.db import models
from django.utils import timezone
import os


class UploadedImage(models.Model):
    id = models.AutoField(primary_key=True)
    original_image = models.ImageField(
        upload_to='uploads/original/%Y/%m/%d/',
        help_text='The original uploaded image file'
    )
    filename = models.CharField(max_length=255, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)
    image_format = models.CharField(max_length=10, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Uploaded Image'
        verbose_name_plural = 'Uploaded Images'
    
    def __str__(self):
        return f"Image {self.id} - {self.filename}"
    
    def save(self, *args, **kwargs):
        if not self.filename and self.original_image:
            self.filename = os.path.basename(self.original_image.name)
        if self.original_image and not self.file_size:
            try:
                from PIL import Image
                self.file_size = self.original_image.size
                self.original_image.seek(0)
                img = Image.open(self.original_image)
                self.width, self.height = img.size
                self.image_format = (img.format or "").upper()
                self.original_image.seek(0)
            except Exception:
                pass
        super().save(*args, **kwargs)


class AnalysisResult(models.Model):
    FORGERY_STATUS_CHOICES = [
        ('pending', 'Pending Analysis'),
        ('authentic', 'Authentic - No Forgery Detected'),
        ('suspicious', 'Suspicious - Possible Forgery'),
        ('forged', 'Forged - Manipulation Detected'),
    ]
    
    image = models.OneToOneField(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name='analysis_result'
    )
    
    forgery_status = models.CharField(
        max_length=20,
        choices=FORGERY_STATUS_CHOICES,
        default='pending'
    )
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        help_text='Confidence score (0-100) of the forgery detection'
    )
    
    ela_image = models.ImageField(
        upload_to='uploads/ela/%Y/%m/%d/',
        null=True,
        blank=True,
        help_text='Error Level Analysis visualization'
    )
    ela_score = models.FloatField(
        null=True,
        blank=True,
        help_text='ELA-based forgery score'
    )
    ela_threshold_exceeded = models.BooleanField(default=False)
    
    heatmap_image = models.ImageField(
        upload_to='uploads/heatmap/%Y/%m/%d/',
        null=True,
        blank=True,
        help_text='Heatmap showing suspicious regions'
    )
    
    analysis_started_at = models.DateTimeField(null=True, blank=True)
    analysis_completed_at = models.DateTimeField(null=True, blank=True)
    processing_time = models.FloatField(
        null=True,
        blank=True,
        help_text='Processing time in seconds'
    )
    
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-analysis_completed_at']
        verbose_name = 'Analysis Result'
        verbose_name_plural = 'Analysis Results'
    
    def __str__(self):
        return f"Analysis Result for Image {self.image.id} - {self.forgery_status}"


class MetadataAnalysis(models.Model):
    analysis_result = models.OneToOneField(
        AnalysisResult,
        on_delete=models.CASCADE,
        related_name='metadata_analysis'
    )
    
    software_detected = models.CharField(max_length=500, blank=True)
    editing_software_found = models.BooleanField(default=False)
    
    camera_make = models.CharField(max_length=100, blank=True)
    camera_model = models.CharField(max_length=100, blank=True)
    
    datetime_original = models.CharField(max_length=100, blank=True)
    datetime_digitized = models.CharField(max_length=100, blank=True)
    datetime_modified = models.CharField(max_length=100, blank=True)
    timestamp_inconsistent = models.BooleanField(default=False)
    
    
    metadata_json = models.TextField(blank=True)
    
    suspicious_indicators = models.TextField(blank=True)
    metadata_score = models.FloatField(
        null=True,
        blank=True,
        help_text='Metadata-based suspicion score (0-100)'
    )
    
    class Meta:
        verbose_name = 'Metadata Analysis'
        verbose_name_plural = 'Metadata Analyses'
    
    def __str__(self):
        return f"Metadata Analysis for Image {self.analysis_result.image.id}"


class SuspiciousRegion(models.Model):
    analysis_result = models.ForeignKey(
        AnalysisResult,
        on_delete=models.CASCADE,
        related_name='suspicious_regions'
    )
    
    x1 = models.FloatField(help_text='Top-left x coordinate')
    y1 = models.FloatField(help_text='Top-left y coordinate')
    x2 = models.FloatField(help_text='Bottom-right x coordinate')
    y2 = models.FloatField(help_text='Bottom-right y coordinate')
    
    confidence = models.FloatField(
        help_text='Confidence level of this region being forged (0-100)'
    )
    area_size = models.IntegerField(
        null=True,
        blank=True,
        help_text='Area in pixels'
    )
    detection_method = models.CharField(
        max_length=50,
        default='ELA',
        help_text='Method used to detect this region'
    )
    
    class Meta:
        ordering = ['-confidence']
        verbose_name = 'Suspicious Region'
        verbose_name_plural = 'Suspicious Regions'
    
    def __str__(self):
        return f"Region ({self.x1}, {self.y1}) - ({self.x2}, {self.y2}) - {self.confidence:.1f}%"
