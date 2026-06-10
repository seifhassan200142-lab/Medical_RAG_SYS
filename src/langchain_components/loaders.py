import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from src.helpers.logger import get_logger
from src.helpers.utils import clean_text

logger = get_logger(__name__)


def load_pdf(path: str) -> list[Document]:
    logger.info(f"Loading PDF: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    cleaned = []
    for doc in docs:
        text = clean_text(doc.page_content)
        if text and len(text) > 20:
            doc.page_content = text
            cleaned.append(doc)
    logger.info(f"Loaded {len(cleaned)} non-empty pages from {path}")
    return cleaned


def load_text(path: str) -> list[Document]:
    logger.info(f"Loading text: {path}")
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def load_markdown(path: str) -> list[Document]:
    logger.info(f"Loading markdown: {path}")
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()
