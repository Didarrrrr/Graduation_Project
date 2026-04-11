"""
Forms for Image Forgery Detection System

Handles user input validation for image uploads.
"""

from django import forms
from .models import UploadedImage


class ImageUploadForm(forms.ModelForm):
    """
    Form for uploading images for forgery detection.
    Validates file type, size, and format.
    """
    
    # Custom file field with styling
    original_image = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/bmp,image/tiff',
            'id': 'image-upload'
        }),
        label='Select Image',
        help_text='Upload JPG, PNG, BMP, or TIFF images (max 10MB)'
    )
    
    class Meta:
        model = UploadedImage
        fields = ['original_image']
    
    def clean_original_image(self):
        """
        Validate the uploaded image file.
        """
        image = self.cleaned_data.get('original_image')
        
        if not image:
            raise forms.ValidationError("Please select an image to upload.")
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10 MB
        if image.size > max_size:
            raise forms.ValidationError(
                f"File size too large. Maximum allowed size is 10 MB. "
                f"Your file is {image.size / (1024 * 1024):.2f} MB."
            )
        
        # Check file extension
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        filename = image.name.lower()
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise forms.ValidationError(
                f"Invalid file format. Allowed formats: JPG, JPEG, PNG, BMP, TIFF."
            )
        
        # Check MIME type
        valid_mime_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
        if hasattr(image, 'content_type') and image.content_type not in valid_mime_types:
            raise forms.ValidationError(
                f"Invalid image type. Please upload a valid image file."
            )
        
        return image


class AnalysisSettingsForm(forms.Form):
    """
    Form for toggling optional analysis modules.
    Core parameters (ELA quality and sensitivity) are fixed in backend settings.
    """

    include_metadata = forms.BooleanField(
        initial=True,
        required=False,
        label='Include Metadata Analysis',
        help_text='Analyze EXIF metadata for editing traces',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    generate_heatmap = forms.BooleanField(
        initial=True,
        required=False,
        label='Generate Heatmap',
        help_text='Create visual heatmap of suspicious regions',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )