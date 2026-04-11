"""
EXIF Metadata Analyzer.

Improved metadata analysis:
- JPEG quantization table analysis for double-compression detection
- EXIF thumbnail consistency check
- Expanded editing software detection with severity tiers
- Resolution/aspect ratio consistency checks
- Better calibrated scoring
"""

import json
import re
import struct
from datetime import datetime
from io import BytesIO

import numpy as np
import piexif
from PIL import Image
from PIL.ExifTags import TAGS


class MetadataAnalyzer:
    """Analyzer for EXIF metadata to detect image manipulation."""

    # Tier 1: Destructive / heavy editors — higher suspicion
    HEAVY_EDITORS = [
        "photoshop", "gimp", "paint.net", "corel", "paintshop",
        "pixlr", "fotor", "luminar", "picsart", "faceapp",
        "remini", "facetune", "meitu", "beautycam", "airbrush",
        "retouch", "inpaint",
    ]

    # Tier 2: Non-destructive / workflow tools — lower suspicion
    LIGHT_EDITORS = [
        "lightroom", "capture one", "darktable", "rawtherapee",
        "affinity", "canva", "snapseed", "vsco", "adobe camera raw",
    ]

    SOCIAL_EXPORT_SOFTWARE = [
        "instagram", "facebook", "whatsapp", "telegram", "wechat", "tiktok",
        "twitter", "snapchat", "line", "signal",
    ]

    SUSPICIOUS_PATTERNS = [
        r"photoshop", r"gimp", r"edited", r"modified", r"retouch", r"composite",
        r"clone", r"heal", r"splice", r"tamper",
    ]

    # Standard JPEG Q-tables (luminance) from the JPEG specification (quality ~75)
    # Used as reference for detecting non-standard quantization
    STANDARD_LUMINANCE_QTABLE = np.array([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ], dtype=np.float64)

    def __init__(self):
        self.suspicious_indicators = []
        self.metadata_dict = {}

    def analyze(self, image_path):
        """Perform comprehensive metadata analysis."""
        self.suspicious_indicators = []
        self.metadata_dict = {}

        try:
            image_format = ""
            actual_size = (0, 0)
            try:
                with Image.open(image_path) as probe:
                    image_format = (probe.format or "").upper()
                    actual_size = probe.size  # (width, height)
            except Exception:
                pass

            exif_data = self._extract_exif(image_path)
            self.metadata_dict["ImageFormat"] = image_format

            software_info = self._analyze_software(exif_data)
            timestamp_info = self._analyze_timestamps(exif_data)
            camera_info = self._analyze_camera(exif_data, image_format=image_format)
            integrity_info = self._analyze_metadata_integrity(exif_data)

            # --- New analyses ---
            qtable_info = self._analyze_quantization_tables(image_path)
            thumbnail_info = self._analyze_thumbnail_consistency(image_path)
            resolution_info = self._analyze_resolution_consistency(exif_data, actual_size)

            metadata_score = self._calculate_metadata_score(
                software_info=software_info,
                timestamp_info=timestamp_info,
                camera_info=camera_info,
                integrity_info=integrity_info,
                image_format=image_format,
                qtable_info=qtable_info,
                thumbnail_info=thumbnail_info,
                resolution_info=resolution_info,
            )
            forgery_indication = self._determine_forgery_indication(metadata_score)

            self.metadata_dict["analysis_summary"] = {
                "software_info": software_info,
                "timestamp_info": timestamp_info,
                "camera_info": camera_info,
                "integrity_info": integrity_info,
                "qtable_info": qtable_info,
                "thumbnail_info": thumbnail_info,
                "resolution_info": resolution_info,
                "metadata_score": metadata_score,
            }

            return {
                "software_detected": software_info.get("software", ""),
                "editing_software_found": software_info.get("is_editing_software", False),
                "camera_make": camera_info.get("make", ""),
                "camera_model": camera_info.get("model", ""),
                "datetime_original": timestamp_info.get("datetime_original", ""),
                "datetime_digitized": timestamp_info.get("datetime_digitized", ""),
                "datetime_modified": timestamp_info.get("datetime_modified", ""),
                "timestamp_inconsistent": timestamp_info.get("inconsistent", False),
                "metadata_json": json.dumps(self.metadata_dict, indent=2, default=str),
                "suspicious_indicators": "\n".join(self.suspicious_indicators),
                "metadata_score": metadata_score,
                "forgery_indication": forgery_indication,
                "full_metadata": self.metadata_dict,
                "metadata_confidence": integrity_info.get("metadata_confidence", 0.0),
            }
        except Exception as e:
            return {
                "error": str(e),
                "software_detected": "",
                "editing_software_found": False,
                "metadata_score": 10,
                "metadata_confidence": 0.0,
                "suspicious_indicators": f"Error analyzing metadata: {str(e)}",
            }
    
    def _extract_exif(self, image_path):
        """Extract EXIF data from image."""
        exif_data = {}

        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                        self.metadata_dict[tag] = str(value)

                self.metadata_dict["Format"] = img.format
                self.metadata_dict["Mode"] = img.mode
                self.metadata_dict["Size"] = img.size
                self.metadata_dict["ICCProfilePresent"] = bool(img.info.get("icc_profile"))
        except Exception as e:
            self.suspicious_indicators.append(f"Could not extract basic EXIF: {str(e)}")

        try:
            exif_dict = piexif.load(image_path)

            for ifd in exif_dict:
                if exif_dict[ifd] is not None:
                    for tag_id, value in exif_dict[ifd].items():
                        try:
                            tag_name = piexif.TAGS[ifd].get(tag_id, {}).get("name", tag_id)
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode("utf-8", errors="ignore")
                                except Exception:
                                    value = str(value)
                            exif_data[f"{ifd}_{tag_name}"] = value
                            self.metadata_dict[f"{ifd}_{tag_name}"] = str(value)
                        except Exception:
                            pass
        except Exception as e:
            self.suspicious_indicators.append(f"Could not extract detailed EXIF: {str(e)}")

        return exif_data

    def _analyze_software(self, exif_data):
        result = {
            "software": "",
            "is_editing_software": False,
            "is_heavy_editor": False,
            "is_light_editor": False,
            "is_social_export": False,
            "is_camera_pipeline": False,
        }

        software_tags = ["Software", "ProcessingSoftware", "CreatorTool", "0_Software"]
        for tag in software_tags:
            if tag in exif_data:
                raw_software = str(exif_data[tag]).strip()
                software = raw_software.lower()
                result["software"] = raw_software

                # Check heavy editors first
                for editor in self.HEAVY_EDITORS:
                    if editor in software:
                        result["is_editing_software"] = True
                        result["is_heavy_editor"] = True
                        self.suspicious_indicators.append(
                            f"Heavy editing software detected: {exif_data[tag]}"
                        )
                        break

                # Check light editors
                if not result["is_heavy_editor"]:
                    for editor in self.LIGHT_EDITORS:
                        if editor in software:
                            result["is_editing_software"] = True
                            result["is_light_editor"] = True
                            self.suspicious_indicators.append(
                                f"Non-destructive editor detected: {exif_data[tag]}"
                            )
                            break

                for social in self.SOCIAL_EXPORT_SOFTWARE:
                    if social in software:
                        result["is_social_export"] = True
                        self.suspicious_indicators.append(
                            f"Social/compression export software detected: {exif_data[tag]}"
                        )
                        break

                camera_pipeline_patterns = ["apple", "samsung", "huawei", "xiaomi", "nikon", "canon", "sony", "fujifilm", "olympus", "panasonic", "leica", "google", "pixel"]
                if any(x in software for x in camera_pipeline_patterns):
                    result["is_camera_pipeline"] = True

                for pattern in self.SUSPICIOUS_PATTERNS:
                    if re.search(pattern, software, re.IGNORECASE):
                        self.suspicious_indicators.append(
                            f"Suspicious software pattern: {exif_data[tag]}"
                        )
                        break
                break

        if "History" in exif_data or "XMP" in str(exif_data):
            self.suspicious_indicators.append("Image has editing history metadata")

        return result

    def _analyze_timestamps(self, exif_data):
        result = {
            "datetime_original": "",
            "datetime_digitized": "",
            "datetime_modified": "",
            "inconsistent": False,
            "large_time_gap": False,
        }

        timestamp_tags = {
            "datetime_original": ["DateTimeOriginal", "DateTime", "0_DateTimeOriginal"],
            "datetime_digitized": ["DateTimeDigitized", "0_DateTimeDigitized"],
            "datetime_modified": ["DateTime", "ModifyDate", "0_DateTime", "FileModifyDate"],
        }

        for key, tags in timestamp_tags.items():
            for tag in tags:
                if tag in exif_data:
                    result[key] = str(exif_data[tag])
                    break

        timestamps = [
            result["datetime_original"],
            result["datetime_digitized"],
            result["datetime_modified"],
        ]
        timestamps = [t for t in timestamps if t]

        parsed_dates = []
        for ts in timestamps:
            dt = self._parse_timestamp(ts)
            if dt is not None:
                parsed_dates.append(dt)
                if dt > datetime.now():
                    self.suspicious_indicators.append(f"Future timestamp detected: {ts}")
                    result["inconsistent"] = True

        # Do not mark as inconsistent for tiny normal differences.
        if len(parsed_dates) > 1:
            seconds_gap = (max(parsed_dates) - min(parsed_dates)).total_seconds()
            if seconds_gap > 120:
                result["inconsistent"] = True
                if seconds_gap > 24 * 3600:
                    result["large_time_gap"] = True
                self.suspicious_indicators.append(
                    f"Timestamp mismatch larger than tolerance: {int(seconds_gap)} seconds"
                )

        return result

    def _analyze_camera(self, exif_data, image_format=""):
        result = {
            "make": "",
            "model": "",
            "camera_profile_strength": 0.0,
        }

        make_tags = ["Make", "0_Make"]
        for tag in make_tags:
            if tag in exif_data:
                result["make"] = str(exif_data[tag]).strip()
                break

        model_tags = ["Model", "0_Model", "CameraModel"]
        for tag in model_tags:
            if tag in exif_data:
                result["model"] = str(exif_data[tag]).strip()
                break

        technical_tags = [
            "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "LensModel",
            "0_ExposureTime", "0_FNumber", "0_ISOSpeedRatings", "0_FocalLength", "Exif_LensModel"
        ]
        technical_count = sum(1 for tag in technical_tags if tag in exif_data)

        has_camera_identity = bool(result["make"] or result["model"])
        if has_camera_identity and technical_count >= 2:
            result["camera_profile_strength"] = 1.0
        elif has_camera_identity or technical_count >= 1:
            result["camera_profile_strength"] = 0.5
        else:
            result["camera_profile_strength"] = 0.0

        if not result["make"] and not result["model"]:
            fmt = str(image_format).upper()
            if fmt in ("PNG", "WEBP", "GIF", "BMP"):
                self.suspicious_indicators.append(
                    "No camera EXIF in file (common for PNG/WebP/screenshots — not strong evidence alone)"
                )
            else:
                self.suspicious_indicators.append(
                    "No camera information found - image may have been processed"
                )

        return result

    def _parse_timestamp(self, ts):
        if not ts:
            return None
        for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"]:
            try:
                return datetime.strptime(ts, fmt).replace(tzinfo=None)
            except ValueError:
                continue
        return None

    def _analyze_metadata_integrity(self, exif_data):
        standard_tags = [
            "DateTimeOriginal", "DateTimeDigitized", "Make", "Model",
            "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength",
        ]
        present = sum(1 for t in standard_tags if t in exif_data or f"0_{t}" in exif_data)
        presence_ratio = present / len(standard_tags)

        jpeg_related = {"0th", "Exif", "GPS", "Interop"}
        has_ifd_content = any(str(k).split("_")[0] in jpeg_related for k in exif_data.keys())
        metadata_confidence = min(1.0, (presence_ratio * 0.8) + (0.2 if has_ifd_content else 0.0))

        if not exif_data:
            self.suspicious_indicators.append("No EXIF blocks found.")
        elif presence_ratio < 0.25:
            self.suspicious_indicators.append("Very sparse EXIF metadata profile.")

        return {
            "presence_ratio": round(presence_ratio, 3),
            "has_ifd_content": has_ifd_content,
            "metadata_confidence": round(metadata_confidence, 3),
        }

    def _analyze_quantization_tables(self, image_path):
        """
        Analyze JPEG quantization tables for signs of double compression.
        
        Double-compressed images (saved twice at different quality levels) have
        characteristic Q-table signatures. Non-standard Q-tables indicate
        recompression at different quality levels.
        
        Returns:
            dict with qtable analysis results
        """
        result = {
            "has_qtable": False,
            "is_standard": True,
            "double_compression_likely": False,
            "qtable_deviation": 0.0,
        }

        try:
            with Image.open(image_path) as img:
                if img.format != "JPEG":
                    return result

                # PIL provides quantization tables for JPEG images
                qtables = img.quantization
                if not qtables:
                    return result

                result["has_qtable"] = True

                # Get the luminance Q-table (table 0)
                if 0 in qtables:
                    lum_table = np.array(qtables[0], dtype=np.float64)

                    # Compare against standard JPEG Q-table
                    # Compute normalized deviation
                    deviation = np.mean(np.abs(lum_table - self.STANDARD_LUMINANCE_QTABLE) /
                                        (self.STANDARD_LUMINANCE_QTABLE + 1e-6))
                    result["qtable_deviation"] = float(deviation)

                    # High deviation from standard suggests non-standard compression
                    if deviation > 0.5:
                        result["is_standard"] = False

                    # Check for double compression signature:
                    # Ratio of Q-values should be approximately constant for single compression
                    # but shows periodic patterns for double compression
                    if len(lum_table) >= 64:
                        ratios = lum_table / (self.STANDARD_LUMINANCE_QTABLE + 1e-6)
                        ratio_std = np.std(ratios)
                        ratio_mean = np.mean(ratios)

                        # High variance in ratios suggests double compression
                        if ratio_mean > 0 and ratio_std / ratio_mean > 0.3:
                            result["double_compression_likely"] = True
                            self.suspicious_indicators.append(
                                "JPEG quantization table suggests possible double compression"
                            )

                # Check if there are multiple Q-tables with unusual properties
                if len(qtables) > 2:
                    self.suspicious_indicators.append(
                        f"Unusual number of JPEG quantization tables: {len(qtables)}"
                    )

        except Exception:
            pass

        return result

    def _analyze_thumbnail_consistency(self, image_path):
        """
        Compare EXIF thumbnail against a downscaled version of the main image.
        Manipulated images often retain the original pre-edit thumbnail,
        creating a mismatch.
        
        Returns:
            dict with thumbnail consistency results
        """
        result = {
            "has_thumbnail": False,
            "thumbnail_mismatch": False,
            "mismatch_score": 0.0,
        }

        try:
            exif_dict = piexif.load(image_path)
            thumbnail_data = exif_dict.get("thumbnail")

            if not thumbnail_data:
                return result

            result["has_thumbnail"] = True

            # Load thumbnail
            thumb_img = Image.open(BytesIO(thumbnail_data)).convert("RGB")
            thumb_array = np.array(thumb_img, dtype=np.float32)

            # Load main image and downscale to thumbnail size
            with Image.open(image_path) as main_img:
                main_rgb = main_img.convert("RGB")
                main_resized = main_rgb.resize(thumb_img.size, Image.LANCZOS)
                main_array = np.array(main_resized, dtype=np.float32)

            # Compare thumbnail vs downscaled main image
            # Use normalized mean absolute error
            diff = np.abs(thumb_array - main_array)
            mae = float(np.mean(diff))

            # Normalize to 0-1 (255 is max possible difference)
            mismatch = mae / 255.0
            result["mismatch_score"] = round(mismatch, 4)

            # Threshold: small differences are expected due to resampling artifacts
            # Score above 0.08 indicates significant content difference
            if mismatch > 0.08:
                result["thumbnail_mismatch"] = True
                self.suspicious_indicators.append(
                    f"EXIF thumbnail differs from main image (score: {mismatch:.3f})"
                )

        except Exception:
            pass

        return result

    def _analyze_resolution_consistency(self, exif_data, actual_size):
        """
        Check if EXIF width/height matches actual image dimensions.
        Mismatches indicate the image was cropped/resized after EXIF was written.
        
        Returns:
            dict with resolution consistency results
        """
        result = {
            "consistent": True,
            "exif_size": None,
            "actual_size": actual_size,
        }

        # Check various EXIF size tags
        exif_width = None
        exif_height = None

        width_tags = ["ExifImageWidth", "ImageWidth", "0_ExifImageWidth", "0_ImageWidth",
                       "Exif_PixelXDimension"]
        height_tags = ["ExifImageHeight", "ImageLength", "0_ExifImageHeight", "0_ImageLength",
                        "Exif_PixelYDimension"]

        for tag in width_tags:
            if tag in exif_data:
                try:
                    exif_width = int(exif_data[tag])
                    break
                except (ValueError, TypeError):
                    pass

        for tag in height_tags:
            if tag in exif_data:
                try:
                    exif_height = int(exif_data[tag])
                    break
                except (ValueError, TypeError):
                    pass

        if exif_width and exif_height and actual_size[0] > 0:
            result["exif_size"] = (exif_width, exif_height)

            # Allow small tolerance (some exporters round dimensions)
            w_match = abs(exif_width - actual_size[0]) <= 2
            h_match = abs(exif_height - actual_size[1]) <= 2

            if not (w_match and h_match):
                result["consistent"] = False
                self.suspicious_indicators.append(
                    f"EXIF dimensions ({exif_width}x{exif_height}) differ from "
                    f"actual ({actual_size[0]}x{actual_size[1]})"
                )

        return result

    def _calculate_metadata_score(
        self, software_info, timestamp_info, camera_info, integrity_info,
        image_format="", qtable_info=None, thumbnail_info=None, resolution_info=None,
    ):
        score = 0.0
        fmt = str(image_format).upper()
        # PNG/WebP often ship without EXIF; down-weight "missing profile" penalties.
        lossless_or_sparse = fmt in ("PNG", "WEBP", "GIF", "BMP")
        camera_penalty_scale = 0.42 if lossless_or_sparse else 1.0
        presence_penalty_scale = 0.55 if lossless_or_sparse else 1.0

        # Software scoring: distinguish heavy vs light editors
        if software_info.get("is_heavy_editor"):
            score += 40  # Heavy editing software (reduced from 45)
        elif software_info.get("is_light_editor"):
            score += 22  # Non-destructive editors are less suspicious
        elif software_info.get("is_social_export"):
            score += 18
        elif software_info.get("software") and not software_info.get("is_camera_pipeline"):
            score += 8

        if timestamp_info.get("inconsistent"):
            score += 16
            if timestamp_info.get("large_time_gap"):
                score += 8

        camera_strength = camera_info.get("camera_profile_strength", 0.0)
        score += (1.0 - camera_strength) * 18 * camera_penalty_scale

        presence_ratio = integrity_info.get("presence_ratio", 0.0)
        score += max(0.0, (0.6 - presence_ratio) * 20) * presence_penalty_scale

        # --- Quantization table analysis ---
        if qtable_info and qtable_info.get("double_compression_likely"):
            score += 20
        elif qtable_info and not qtable_info.get("is_standard"):
            score += 8

        # --- Thumbnail consistency ---
        if thumbnail_info and thumbnail_info.get("thumbnail_mismatch"):
            mismatch = thumbnail_info.get("mismatch_score", 0)
            if mismatch > 0.15:
                score += 25  # Strong mismatch
            elif mismatch > 0.08:
                score += 15  # Moderate mismatch

        # --- Resolution consistency ---
        if resolution_info and not resolution_info.get("consistent", True):
            score += 10

        indicator_count = len(self.suspicious_indicators)
        score += min(12, indicator_count * 2.0)

        return min(100.0, round(score, 2))

    def _determine_forgery_indication(self, score):
        if score >= 60:
            return "High - Strong indicators of manipulation"
        elif score >= 40:
            return "Medium - Some suspicious indicators present"
        elif score >= 20:
            return "Low - Minor anomalies detected"
        else:
            return "None - No suspicious metadata indicators"


def analyze_metadata(image_path):
    """Convenience function to analyze image metadata."""
    analyzer = MetadataAnalyzer()
    return analyzer.analyze(image_path)