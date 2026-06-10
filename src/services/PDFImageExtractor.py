"""
PDF Image Extractor
Extracts images and tables from PDF pages using PyMuPDF (fitz).
Every extracted image is passed to MedicalImageService for full analysis.
"""
import io
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from src.helpers.logger import get_logger

logger = get_logger(__name__)

MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80
MIN_IMAGE_BYTES = 4096


@dataclass
class ExtractedImage:
    image_id: str
    page_number: int
    image_index: int
    pil_image: object           # PIL.Image.Image
    width: int
    height: int
    colorspace: str
    xref: int
    near_text: str = ""         # surrounding text on same page
    caption: str = ""           # text immediately below the image bbox


@dataclass
class ExtractedTable:
    table_id: str
    page_number: int
    table_index: int
    markdown_text: str
    raw_text: str
    bbox: tuple


class PDFImageExtractor:
    """
    Extracts images and tables from every page of a PDF using PyMuPDF.
    Returns structured ExtractedImage and ExtractedTable objects.
    """

    def extract(self, pdf_path: str) -> tuple[list[ExtractedImage], list[ExtractedTable]]:
        images: list[ExtractedImage] = []
        tables: list[ExtractedTable] = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF for image extraction: {e}")
            return images, tables

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = self._extract_page_images(doc, page, page_num)
            images.extend(page_images)
            page_tables = self._extract_page_tables(page, page_num)
            tables.extend(page_tables)

        doc.close()
        logger.info(
            f"PDF extraction complete: {len(images)} images, {len(tables)} tables "
            f"from {pdf_path}"
        )
        return images, tables

    def _extract_page_images(
        self, doc: fitz.Document, page: fitz.Page, page_num: int
    ) -> list[ExtractedImage]:
        extracted = []
        page_text = page.get_text("text")
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.debug(f"Failed to extract image xref={xref} page={page_num}: {e}")
                continue

            if not base_image:
                continue

            img_bytes = base_image.get("image", b"")
            if len(img_bytes) < MIN_IMAGE_BYTES:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(img_bytes))
                if pil_img.mode not in ("RGB", "RGBA", "L", "P"):
                    pil_img = pil_img.convert("RGB")
                elif pil_img.mode == "P":
                    pil_img = pil_img.convert("RGB")
            except Exception as e:
                logger.debug(f"Failed to open image bytes page={page_num}: {e}")
                continue

            caption = self._find_image_caption(page, img_info, page_text)
            near_text = self._get_near_text(page_text, max_chars=500)

            extracted.append(ExtractedImage(
                image_id=str(uuid.uuid4()),
                page_number=page_num + 1,
                image_index=img_idx,
                pil_image=pil_img,
                width=width,
                height=height,
                colorspace=base_image.get("colorspace", 0),
                xref=xref,
                near_text=near_text,
                caption=caption,
            ))

        return extracted

    def _find_image_caption(
        self, page: fitz.Page, img_info: tuple, page_text: str
    ) -> str:
        try:
            img_rects = page.get_image_rects(img_info[0])
            if not img_rects:
                return ""
            rect = img_rects[0]
            # Search for text below the image bounding box
            below_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 60)
            below_text = page.get_text("text", clip=below_rect).strip()
            if below_text and len(below_text) > 5:
                return below_text[:300]
        except Exception:
            pass
        return ""

    def _get_near_text(self, page_text: str, max_chars: int = 500) -> str:
        return page_text[:max_chars].strip()

    def _extract_page_tables(self, page: fitz.Page, page_num: int) -> list[ExtractedTable]:
        tables = []
        try:
            found_tables = page.find_tables()
            if not found_tables or not found_tables.tables:
                return tables
            for t_idx, table in enumerate(found_tables.tables):
                try:
                    df = table.to_pandas()
                    if df.empty:
                        continue
                    markdown_text = df.to_markdown(index=False)
                    raw_text = df.to_string(index=False)
                    bbox = tuple(table.bbox) if hasattr(table, "bbox") else (0, 0, 0, 0)
                    tables.append(ExtractedTable(
                        table_id=str(uuid.uuid4()),
                        page_number=page_num + 1,
                        table_index=t_idx,
                        markdown_text=markdown_text or raw_text,
                        raw_text=raw_text,
                        bbox=bbox,
                    ))
                except Exception as e:
                    logger.debug(f"Table extraction failed page={page_num} idx={t_idx}: {e}")
        except Exception as e:
            logger.debug(f"Table detection failed page={page_num}: {e}")
        return tables
