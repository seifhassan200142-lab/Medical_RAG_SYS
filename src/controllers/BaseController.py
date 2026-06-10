from src.helpers.exceptions import RAGException
from src.helpers.logger import get_logger

logger = get_logger(__name__)


class BaseController:
    def handle_exception(self, exc: Exception):
        if isinstance(exc, RAGException):
            logger.error(f"{type(exc).__name__}: {exc}")
            raise exc
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise RAGException(str(exc)) from exc
