import hashlib
import re
import uuid
from datetime import datetime, timezone


def generate_id() -> str:
    return str(uuid.uuid4())

def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def sanitize_filename(filename: str) -> str:
    filename = re.sub(r"[^\w.\-]", "_", filename)
    return filename[:200]

def truncate(text: str, max_chars: int = 200) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

def extract_year(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group()) if match else None

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
