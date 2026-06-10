from pydantic import BaseModel, Field
from typing import Optional, Any


class ImageFinding(BaseModel):
    finding_type: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    region: str = "unspecified"
    severity: str = "info"  # info | low | moderate | high | critical


class ImageReport(BaseModel):
    modality: str
    overall_severity: str
    overall_confidence: float
    impression: str
    findings_summary: str
    recommendations: list[str]
    critical_flags: list[str]
    report_date: str


class MedicalImageModel(BaseModel):
    image_id: str
    filename: str
    stored_path: str
    modality: str
    checksum: str
    upload_date: str
    width: int
    height: int
    channels: int
    file_format: str
    page_number: int = 0
    clinical_context: Optional[str] = None
    ocr_text: Optional[str] = None
    caption: Optional[str] = None
    findings: list[ImageFinding] = []
    report: ImageReport
    metadata: dict[str, Any] = {}
    base64_thumbnail: Optional[str] = None
