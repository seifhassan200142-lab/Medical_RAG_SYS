"""
Medical Image Service
Handles preprocessing, analysis, findings extraction, OCR, VLM analysis,
caption generation, report generation, and metadata for medical images
extracted from PDFs (X-Ray, CT, MRI, Ultrasound, Pathology, Clinical, Charts).

No longer handles file-system ingestion from HTTP uploads.
Operates on PIL.Image objects passed directly from PDFImageExtractor.
"""
import io
import os
import base64
import uuid
from datetime import datetime, timezone
from typing import Optional, Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

from src.helpers.config import settings
from src.helpers.logger import get_logger
from src.models.MedicalImageModel import MedicalImageModel, ImageFinding, ImageReport

logger = get_logger(__name__)

IMAGE_MODALITY_SIGNATURES = {
    "xray": [
        "x-ray", "xray", "x ray", "radiograph", "chest pa", "chest ap",
        "plain film", "kv", "mas", "bone", "lung", "rib", "spine",
        "anteroposterior", "posteroanterior",
    ],
    "ct": [
        "ct scan", "computed tomography", "computed tomograph", "hounsfield",
        "hu ", "axial", "coronal", "sagittal", "contrast enhanced", "non-contrast",
        "slice thickness", "window width", "window level", "helical",
    ],
    "mri": [
        "mri", "magnetic resonance", "t1", "t2", "flair", "dwi", "adc",
        "t1w", "t2w", "gadolinium", "tesla", "tr ", "te ", "field strength",
        "proton density", "bold", "diffusion weighted",
    ],
    "ultrasound": [
        "ultrasound", "sonogram", "sonography", "echogenicity", "hyperechoic",
        "hypoechoic", "anechoic", "doppler", "transducer", "mhz", "probe",
        "acoustic shadow", "echocardiography",
    ],
    "pathology": [
        "pathology", "histology", "biopsy", "hematoxylin", "eosin", "h&e",
        "stain", "magnification", "cells", "tissue", "nucleus", "cytology",
        "immunohistochemistry", "ihc", "microscopy", "section",
    ],
    "chart": [
        "figure", "fig.", "graph", "chart", "plot", "diagram", "curve",
        "kaplan", "survival", "forest plot", "odds ratio", "hazard",
        "bar graph", "histogram", "scatter",
    ],
    "clinical": [
        "clinical photo", "wound", "skin", "lesion", "rash", "swelling",
        "erythema", "dermato", "photograph", "clinical image", "patient photo",
    ],
}

MODALITY_PREPROCESSING = {
    "xray":      {"enhance_contrast": True,  "denoise": True,  "normalize": True,  "clahe": True},
    "ct":        {"enhance_contrast": False, "denoise": True,  "normalize": True,  "clahe": True},
    "mri":       {"enhance_contrast": False, "denoise": True,  "normalize": True,  "clahe": False},
    "ultrasound":{"enhance_contrast": True,  "denoise": True,  "normalize": True,  "clahe": False},
    "pathology": {"enhance_contrast": True,  "denoise": False, "normalize": False, "clahe": False},
    "chart":     {"enhance_contrast": False, "denoise": False, "normalize": False, "clahe": False},
    "clinical":  {"enhance_contrast": True,  "denoise": False, "normalize": False, "clahe": False},
}

ANATOMICAL_REGIONS = {
    "xray":      ["chest", "abdomen", "pelvis", "spine", "extremity", "skull", "knee", "hip", "shoulder", "wrist"],
    "ct":        ["head", "chest", "abdomen", "pelvis", "spine", "neck", "extremity", "whole body"],
    "mri":       ["brain", "spine", "knee", "shoulder", "hip", "abdomen", "pelvis", "breast", "heart"],
    "ultrasound":["abdomen", "pelvis", "thyroid", "breast", "cardiac", "vascular", "obstetric", "renal"],
    "pathology": ["lung", "colon", "breast", "prostate", "liver", "kidney", "skin", "lymph node", "brain"],
    "chart":     ["statistical", "survival", "clinical outcome", "treatment comparison"],
    "clinical":  ["skin", "wound", "extremity", "face", "oral", "eye", "ear", "neck"],
}


class MedicalImageService:
    def __init__(self, image_store_path: str = "uploads/images"):
        self.image_store_path = image_store_path
        os.makedirs(image_store_path, exist_ok=True)
        self._vlm_available = self._check_vlm_availability()

    def _check_vlm_availability(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))

    def analyze_pil_image(
        self,
        pil_image: Image.Image,
        image_id: str,
        original_filename: str,
        page_number: int,
        modality_hint: Optional[str] = None,
        context_text: str = "",
        caption: str = "",
    ) -> MedicalImageModel:
        """
        Full analysis pipeline for a PIL image extracted from a PDF.
        Returns a MedicalImageModel with findings, OCR text, caption, and report.
        """
        if pil_image.mode not in ("RGB", "L", "RGBA"):
            pil_image = pil_image.convert("RGB")

        modality = self._detect_modality(pil_image, original_filename, modality_hint, context_text + " " + caption)
        preprocessed = self._preprocess_image(pil_image, modality)
        base64_image = self._image_to_base64(preprocessed)

        ocr_text = self._run_ocr(preprocessed)
        combined_context = " ".join(filter(None, [context_text, caption, ocr_text]))

        raw_metadata = self._extract_pil_metadata(pil_image, modality, page_number)
        findings = self._extract_findings(preprocessed, modality, combined_context, base64_image)
        generated_caption = self._generate_caption(findings, modality, caption, ocr_text)
        report = self._generate_report(findings, modality, combined_context)

        stored_path = self._store_preprocessed(preprocessed, image_id, modality)

        return MedicalImageModel(
            image_id=image_id,
            filename=original_filename,
            stored_path=stored_path,
            modality=modality,
            checksum="",
            upload_date=datetime.now(timezone.utc).isoformat(),
            width=preprocessed.width,
            height=preprocessed.height,
            channels=len(preprocessed.getbands()),
            file_format="JPEG",
            clinical_context=combined_context[:500] if combined_context else None,
            ocr_text=ocr_text,
            caption=generated_caption,
            page_number=page_number,
            findings=findings,
            report=report,
            metadata=raw_metadata,
            base64_thumbnail=self._create_thumbnail_b64(pil_image),
        )

    def _detect_modality(
        self,
        image: Image.Image,
        filename: str,
        hint: Optional[str],
        context: str,
    ) -> str:
        if hint and hint.lower() in IMAGE_MODALITY_SIGNATURES:
            return hint.lower()

        combined_text = (filename + " " + context).lower()
        scores = {m: 0 for m in IMAGE_MODALITY_SIGNATURES}
        for modality, keywords in IMAGE_MODALITY_SIGNATURES.items():
            for kw in keywords:
                if kw in combined_text:
                    scores[modality] += 1

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

        return self._detect_modality_from_pixels(image)

    def _detect_modality_from_pixels(self, image: Image.Image) -> str:
        arr = np.array(image.convert("RGB"))
        grayscale_ratio = self._compute_grayscale_ratio(arr)
        mean_brightness = float(np.mean(arr))

        if grayscale_ratio > 0.92:
            std_brightness = float(np.std(arr))
            if mean_brightness < 80 and std_brightness > 40:
                return "xray"
            if std_brightness < 30:
                return "mri"
            return "ct"

        color_saturation = self._compute_saturation(arr)
        if color_saturation > 0.3 and mean_brightness > 100:
            return "pathology"
        return "clinical"

    def _compute_grayscale_ratio(self, rgb_array: np.ndarray) -> float:
        r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
        diff = np.abs(r.astype(int) - g.astype(int)) + np.abs(g.astype(int) - b.astype(int))
        return float(np.mean(diff < 10))

    def _compute_saturation(self, rgb_array: np.ndarray) -> float:
        max_c = rgb_array.max(axis=2).astype(float)
        min_c = rgb_array.min(axis=2).astype(float)
        sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0)
        return float(np.mean(sat))

    def _preprocess_image(self, image: Image.Image, modality: str) -> Image.Image:
        config = MODALITY_PREPROCESSING.get(modality, MODALITY_PREPROCESSING["clinical"])
        processed = image.copy()
        if processed.mode == "RGBA":
            processed = processed.convert("RGB")

        if config["normalize"]:
            processed = self._normalize_image(processed)
        if config["clahe"]:
            processed = self._apply_clahe(processed)
        if config["denoise"]:
            processed = processed.filter(ImageFilter.MedianFilter(size=3))
        if config["enhance_contrast"]:
            processed = ImageEnhance.Contrast(processed).enhance(1.3)

        max_dim = 1024
        if max(processed.width, processed.height) > max_dim:
            processed.thumbnail((max_dim, max_dim), Image.LANCZOS)

        return processed

    def _normalize_image(self, image: Image.Image) -> Image.Image:
        arr = np.array(image).astype(float)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255
        return Image.fromarray(arr.astype(np.uint8))

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        if image.mode == "L":
            arr = np.array(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return Image.fromarray(clahe.apply(arr))
        else:
            lab = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            enhanced = cv2.merge((l_channel, a, b))
            return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        fmt = "JPEG" if image.mode == "RGB" else "PNG"
        image.save(buffer, format=fmt, quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _create_thumbnail_b64(self, image: Image.Image, size: tuple = (256, 256)) -> str:
        thumb = image.copy()
        thumb.thumbnail(size, Image.LANCZOS)
        if thumb.mode not in ("RGB", "L"):
            thumb = thumb.convert("RGB")
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _run_ocr(self, image: Image.Image) -> str:
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, lang="eng")
            text = text.strip()
            if len(text) < 10:
                return ""
            return text[:2000]
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return ""

    def _extract_findings(
        self,
        image: Image.Image,
        modality: str,
        context: str,
        base64_image: str,
    ) -> list[ImageFinding]:
        findings = list(self._analyze_pixel_characteristics(image, modality))

        if self._vlm_available:
            try:
                vlm_findings = self._run_vlm_analysis(base64_image, modality, context)
                findings.extend(vlm_findings)
            except Exception as e:
                logger.warning(f"VLM analysis failed: {e}")

        return findings

    def _analyze_pixel_characteristics(self, image: Image.Image, modality: str) -> list[ImageFinding]:
        findings = []
        arr = np.array(image.convert("L"))
        mean_intensity = float(np.mean(arr))
        std_intensity = float(np.std(arr))

        if modality == "xray":
            opacity_ratio = float(np.mean(arr > 180)) * 100
            if opacity_ratio > 15:
                findings.append(ImageFinding(
                    finding_type="opacity",
                    description=f"Increased opacity detected in {opacity_ratio:.1f}% of image area",
                    confidence=0.6,
                    region="detected_area",
                    severity="moderate" if opacity_ratio > 40 else "low",
                ))

        if modality in ("ct", "mri"):
            high_ratio = float(np.mean(arr > 200)) * 100
            low_ratio = float(np.mean(arr < 30)) * 100
            if high_ratio > 5:
                findings.append(ImageFinding(
                    finding_type="hyperintense_region",
                    description=f"Hyperintense regions detected: {high_ratio:.1f}% of scan area",
                    confidence=0.55,
                    region="detected_area",
                    severity="moderate",
                ))
            if low_ratio > 30:
                findings.append(ImageFinding(
                    finding_type="hypointense_region",
                    description=f"Hypointense regions: {low_ratio:.1f}% — possible fluid or necrosis",
                    confidence=0.5,
                    region="detected_area",
                    severity="low",
                ))

        if modality == "ultrasound":
            total = arr.size
            anechoic = float(np.sum(arr < 30) / total * 100)
            if anechoic > 15:
                findings.append(ImageFinding(
                    finding_type="anechoic_region",
                    description=f"Anechoic (fluid-filled) region: {anechoic:.1f}% of image",
                    confidence=0.65,
                    region="detected_area",
                    severity="low",
                ))

        if modality == "pathology":
            rgb = np.array(image.convert("RGB"))
            r, g, b = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)
            purple_ratio = float(np.mean((r > 80) & (r < 200) & (b > 80) & (b < 220) & (g < 150)) * 100)
            if purple_ratio > 5:
                findings.append(ImageFinding(
                    finding_type="hematoxylin_staining",
                    description=f"Nuclear hematoxylin staining: {purple_ratio:.1f}% coverage",
                    confidence=0.7,
                    region="global",
                    severity="info",
                ))

        edge_density = float(np.mean(cv2.Canny(arr, 50, 150) > 0))
        findings.append(ImageFinding(
            finding_type="image_quality",
            description=(
                f"Image quality — sharpness: {edge_density:.3f} | "
                f"mean intensity: {mean_intensity:.1f} | std: {std_intensity:.1f}"
            ),
            confidence=1.0,
            region="global",
            severity="info",
        ))
        return findings

    def _run_vlm_analysis(
        self,
        base64_image: str,
        modality: str,
        context: str,
    ) -> list[ImageFinding]:
        import json
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        modality_instructions = {
            "xray": "Focus on: lung fields, cardiac silhouette, bony structures, diaphragm, pleural spaces, mediastinum. Note opacities, consolidations, effusions, pneumothorax, cardiomegaly, fractures.",
            "ct": "Focus on: tissue densities, lesion characterization (size, margins, enhancement), vascular structures, organ morphology, lymph nodes. Note masses, hemorrhage, infarcts, edema.",
            "mri": "Focus on: T1/T2 signal characteristics, lesion morphology, enhancement patterns, diffusion restriction. Note abnormal signal, mass effect, structural anomalies.",
            "ultrasound": "Focus on: echogenicity patterns, posterior acoustic features, vascularity, organ measurements, cystic vs solid lesions. Note masses, free fluid, calcifications.",
            "pathology": "Focus on: cellular architecture, nuclear morphology, staining patterns, necrosis, mitoses, inflammatory infiltrates. Note dysplasia or malignancy features.",
            "chart": "Extract key data points, trends, statistical values, axis labels, and clinical significance from this medical chart or graph.",
            "clinical": "Focus on: lesion characteristics (color, borders, elevation, distribution), tissue appearance, surrounding changes.",
        }.get(modality, "Describe all visible medical findings systematically.")

        context_note = f"\nClinical context: {context[:300]}" if context else ""
        prompt = (
            f"You are a radiologist/pathologist analyzing a medical image extracted from a PDF.\n"
            f"Modality: {modality.upper()}{context_note}\n\n"
            f"{modality_instructions}\n\n"
            "Return a JSON array of findings:\n"
            '[{"finding_type":"...","description":"...","confidence":0.0-1.0,"region":"...","severity":"info|low|moderate|high|critical"}]\n'
            "Return ONLY valid JSON. No preamble or explanation."
        )

        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
            temperature=0.1,
        )

        raw = (response.choices[0].message.content or "[]").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return [
            ImageFinding(
                finding_type=f.get("finding_type", "observation"),
                description=f.get("description", ""),
                confidence=float(f.get("confidence", 0.5)),
                region=f.get("region", "unspecified"),
                severity=f.get("severity", "info"),
            )
            for f in parsed
            if isinstance(f, dict) and f.get("description")
        ]

    def _generate_caption(
        self,
        findings: list[ImageFinding],
        modality: str,
        original_caption: str,
        ocr_text: str,
    ) -> str:
        if original_caption and len(original_caption) > 20:
            return original_caption

        non_quality = [f for f in findings if f.finding_type != "image_quality"]
        if non_quality:
            top = non_quality[0]
            return f"{modality.upper()} image — {top.description}"

        if ocr_text and len(ocr_text) > 20:
            return ocr_text[:200]

        return f"{modality.upper()} medical image"

    def _generate_report(
        self,
        findings: list[ImageFinding],
        modality: str,
        context: str,
    ) -> ImageReport:
        critical = [f for f in findings if f.severity == "critical"]
        high = [f for f in findings if f.severity == "high"]
        moderate = [f for f in findings if f.severity == "moderate"]

        if critical:
            overall_severity = "critical"
        elif high:
            overall_severity = "high"
        elif moderate:
            overall_severity = "moderate"
        else:
            overall_severity = "low"

        overall_confidence = round(
            sum(f.confidence for f in findings) / len(findings), 3
        ) if findings else 0.0

        non_quality = [f for f in findings if f.finding_type != "image_quality"]
        if non_quality:
            impression_lines = [f"{modality.upper()} demonstrates:"]
            for f in non_quality[:5]:
                impression_lines.append(f"- {f.description}")
        else:
            impression_lines = [f"{modality.upper()} image — no significant automated findings."]

        impression_lines.append("Note: Automated analysis only. Clinical correlation required.")

        recs = []
        if overall_severity in ("critical", "high"):
            recs.append("Priority radiologist review recommended.")
        else:
            recs.append("Routine review recommended.")
        recs.append("Correlate with clinical history and other findings.")

        return ImageReport(
            modality=modality,
            overall_severity=overall_severity,
            overall_confidence=overall_confidence,
            impression="\n".join(impression_lines),
            findings_summary=" | ".join(f.description for f in non_quality[:3]) or "No findings.",
            recommendations=recs,
            critical_flags=[f.description for f in critical],
            report_date=datetime.now(timezone.utc).isoformat(),
        )

    def _extract_pil_metadata(
        self,
        image: Image.Image,
        modality: str,
        page_number: int,
    ) -> dict[str, Any]:
        arr = np.array(image.convert("L"))
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "modality": modality,
            "page_number": page_number,
            "pixel_count": image.width * image.height,
            "aspect_ratio": round(image.width / image.height, 3) if image.height else 0,
            "mean_pixel_intensity": round(float(np.mean(arr)), 2),
            "std_pixel_intensity": round(float(np.std(arr)), 2),
            "possible_anatomical_regions": ANATOMICAL_REGIONS.get(modality, []),
        }

    def _store_preprocessed(self, image: Image.Image, image_id: str, modality: str) -> str:
        filename = f"{image_id}_{modality}.jpg"
        path = os.path.join(self.image_store_path, filename)
        img_to_save = image if image.mode == "RGB" else image.convert("RGB")
        img_to_save.save(path, format="JPEG", quality=90)
        return path

    def get_image_as_text(self, model: MedicalImageModel) -> str:
        lines = [
            f"[MEDICAL IMAGE] Source: {model.filename} | Page: {model.page_number} | Modality: {model.modality.upper()}",
            f"Caption: {model.caption}" if model.caption else "",
            f"Dimensions: {model.width}x{model.height}",
        ]
        if model.ocr_text:
            lines.append(f"OCR Text: {model.ocr_text[:500]}")
        if model.clinical_context:
            lines.append(f"Context: {model.clinical_context[:300]}")

        lines.append("\nFINDINGS:")
        for i, finding in enumerate(model.findings, 1):
            if finding.finding_type == "image_quality":
                continue
            lines.append(f"  [{i}] [{finding.severity.upper()}] {finding.description} (confidence: {finding.confidence:.2f})")

        lines.append(f"\nIMPRESSION:\n{model.report.impression}")

        if model.report.critical_flags:
            lines.append("\n⚠ CRITICAL:")
            for flag in model.report.critical_flags:
                lines.append(f"  !! {flag}")

        return "\n".join(line for line in lines if line)
