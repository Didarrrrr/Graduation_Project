from __future__ import annotations

import os
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import MetadataAnalysis

C_BG = colors.HexColor("#FFFFFF")
C_MUTED = colors.HexColor("#64748B")
C_BODY = colors.HexColor("#1E293B")
C_BORDER = colors.HexColor("#E2E8F0")
C_PRIMARY = colors.HexColor("#4F6EF7")
C_CLEAN = colors.HexColor("#059669")
C_FORGERY = colors.HexColor("#E8550A")
C_UNCERTAIN = colors.HexColor("#D97706")


def _safe_metadata(result):
    try:
        return result.metadata_analysis
    except MetadataAnalysis.DoesNotExist:
        return None


def _escape(s) -> str:
    if s is None:
        return ""
    t = str(s)
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_hex(c: colors.Color) -> str:
    return "#%06x" % (c.int_rgb() & 0xFFFFFF)


def _status_colors(status: str):
    if status == "forged":
        return C_FORGERY, colors.HexColor("#FFF4ED")
    if status == "suspicious":
        return C_UNCERTAIN, colors.HexColor("#FFFBEB")
    if status == "authentic":
        return C_CLEAN, colors.HexColor("#ECFDF5")
    return C_MUTED, colors.HexColor("#F8FAFC")


def build_analysis_report_pdf(result, component_scores: dict) -> BytesIO:
    buffer = BytesIO()
    meta = _safe_metadata(result)
    accent, verdict_bg = _status_colors(result.forgery_status or "")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=C_BODY,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportMuted",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=C_MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=C_BODY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportVerdict",
            parent=styles["Normal"],
            fontSize=14,
            leading=18,
            textColor=accent,
            fontName="Helvetica-Bold",
        )
    )

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Forgery analysis summary",
    )
    story = []

    story.append(Paragraph("Image forgery analysis — summary", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "<font color='%s'>Multi-method screening (ELA, metadata, copy-move, noise)</font>"
            % _html_hex(C_MUTED),
            styles["ReportMuted"],
        )
    )
    gen = result.analysis_completed_at
    gen_s = gen.strftime("%Y-%m-%d %H:%M UTC") if gen else "—"
    story.append(Paragraph(_escape(f"Generated: {gen_s}"), styles["ReportMuted"]))
    story.append(Spacer(1, 14))

    status_display = result.get_forgery_status_display()
    conf = float(result.confidence_score or 0.0)
    verdict_box = Table(
        [
            [
                Paragraph(_escape(status_display), styles["ReportVerdict"]),
                Paragraph(
                    f"<font size='22' color='{_html_hex(accent)}'><b>{conf:.1f}%</b></font>"
                    f"<br/><font size='8' color='{_html_hex(C_MUTED)}'>Fused confidence</font>",
                    styles["ReportBody"],
                ),
            ],
        ],
        colWidths=[3.2 * inch, 3.3 * inch],
    )
    verdict_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
                ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(verdict_box)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "<b>File:</b> %s<br/>"
            "<b>Analysis ID:</b> %s · <b>Runtime:</b> %.2fs · <b>ELA threshold exceeded:</b> %s"
            % (
                _escape(result.image.filename),
                result.image.id,
                float(result.processing_time or 0),
                "Yes" if result.ela_threshold_exceeded else "No",
            ),
            styles["ReportBody"],
        )
    )
    story.append(Spacer(1, 14))

    thumb_path = None
    try:
        if result.image.original_image:
            thumb_path = result.image.original_image.path
    except Exception:
        thumb_path = None
    if thumb_path and os.path.isfile(thumb_path):
        try:
            from PIL import Image as PILImage

            im = PILImage.open(thumb_path)
            iw, ih = im.size
            max_w = 2.2 * inch
            scale = min(1.0, max_w / float(iw))
            rw = iw * scale
            rh = ih * scale
            story.append(Paragraph("<b>Image</b>", styles["ReportBody"]))
            story.append(Spacer(1, 4))
            story.append(RLImage(thumb_path, width=rw, height=rh, kind="proportional"))
            story.append(Spacer(1, 12))
        except Exception:
            pass

    score_rows = [
        ["Method", "Score"],
        ["ELA", f"{float(result.ela_score or 0):.1f}"],
        [
            "Copy-move",
            f"{float(component_scores.get('copy_move_score', 0) or 0):.1f}",
        ],
        ["Noise", f"{float(component_scores.get('noise_score', 0) or 0):.1f}"],
    ]
    if meta is not None:
        score_rows.append(
            ["Metadata", f"{float(meta.metadata_score or 0):.1f}"],
        )

    score_tbl = Table(score_rows, colWidths=[3.5 * inch, 1.5 * inch])
    score_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Paragraph("<b>Method scores</b> (0–100 scale)", styles["ReportBody"]))
    story.append(Spacer(1, 6))
    story.append(score_tbl)
    story.append(Spacer(1, 14))

    if meta is not None:
        story.append(Paragraph("<b>Metadata (short)</b>", styles["ReportBody"]))
        story.append(Spacer(1, 4))
        meta_line = (
            f"Software: {_escape(meta.software_detected) or '—'} · "
            f"Editor flagged: {'Yes' if meta.editing_software_found else 'No'}<br/>"
            f"Camera: {_escape((meta.camera_make or '').strip())} {_escape((meta.camera_model or '').strip())} · "
            f"Taken: {_escape(meta.datetime_original) or '—'} · "
            f"Time inconsistent: {'Yes' if meta.timestamp_inconsistent else 'No'}"
        )
        story.append(Paragraph(meta_line, styles["ReportBody"]))
        story.append(Spacer(1, 12))

    regions = list(result.suspicious_regions.all())
    if regions:
        story.append(Paragraph("<b>Suspicious regions</b> (top 12)", styles["ReportBody"]))
        story.append(Spacer(1, 6))
        reg_data = [["#", "Method", "Conf. %"]]
        for i, r in enumerate(regions[:12], 1):
            reg_data.append(
                [
                    str(i),
                    _escape(r.detection_method),
                    f"{float(r.confidence):.1f}",
                ]
            )
        reg_tbl = Table(reg_data, colWidths=[0.45 * inch, 2.2 * inch, 1.2 * inch])
        reg_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, colors.HexColor("#F8FAFC")]),
                    ("GRID", (0, 0), (-1, -1), 0.25, C_BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(reg_tbl)
        if len(regions) > 12:
            story.append(
                Spacer(1, 4),
            )
            story.append(
                Paragraph(
                    _escape(f"… and {len(regions) - 12} more (see web result for full list)."),
                    styles["ReportMuted"],
                )
            )
        story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "<i>Educational screening only — not legal evidence. Scores are heuristic.</i>",
            styles["ReportMuted"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer
