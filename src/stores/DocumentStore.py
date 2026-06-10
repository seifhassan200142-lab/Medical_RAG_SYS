from src.models.DocumentModel import DocumentModel
from src.helpers.logger import get_logger

logger = get_logger(__name__)


class DocumentStore:
    def __init__(self):
        self._documents: dict[str, DocumentModel] = {}

    def save(self, doc: DocumentModel) -> None:
        self._documents[doc.id] = doc
        logger.info(f"Stored document: id={doc.id} type={doc.document_type} file={doc.filename}")

    def get(self, document_id: str) -> DocumentModel | None:
        return self._documents.get(document_id)

    def all(self) -> list[DocumentModel]:
        return list(self._documents.values())

    def exists_by_checksum(self, checksum: str) -> DocumentModel | None:
        for doc in self._documents.values():
            if doc.checksum == checksum:
                return doc
        return None

    def delete(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False

    def count(self) -> int:
        return len(self._documents)

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self._documents.values():
            counts[doc.document_type] = counts.get(doc.document_type, 0) + 1
        return counts
