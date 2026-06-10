class RAGException(Exception):
    pass

class EmbeddingError(RAGException):
    pass

class IngestionError(RAGException):
    pass

class UnsupportedFileTypeError(IngestionError):
    def __init__(self, ext: str):
        super().__init__(
            f"Unsupported file type: '.{ext}'. Supported: pdf, txt, md"
        )

class RetrievalError(RAGException):
    pass

class GenerationError(RAGException):
    pass

class StorageError(RAGException):
    pass

class HyDEError(RAGException):
    pass

class ConfidenceError(RAGException):
    pass

class MetadataExtractionError(RAGException):
    pass

class ImageProcessingError(RAGException):
    pass

class RerankingError(RAGException):
    pass

class ValidationError(RAGException):
    pass
