class DocumentError(Exception):
    """Base exception for document ingestion errors."""


class EmptyFileError(DocumentError):
    """Raised when an uploaded file has zero bytes."""


class FileTooLargeError(DocumentError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        super().__init__(f"File exceeds max size of {max_bytes} bytes.")
