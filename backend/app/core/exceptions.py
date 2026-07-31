class DocumentError(Exception):
    """Base exception for document ingestion errors."""


class EmptyFileError(DocumentError):
    """Raised when an uploaded file has zero bytes."""


class FileTooLargeError(DocumentError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        super().__init__(f"File exceeds max size of {max_bytes} bytes.")


class DocumentNotFoundError(DocumentError):
    """Raised when no stored file matches the given document id."""

    def __init__(self, document_id: object):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' was not found.")


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when no parser is registered for a file's extension."""

    def __init__(self, extension: str):
        self.extension = extension
        super().__init__(f"No parser registered for extension '{extension}'.")


class ParsingError(DocumentError):
    """Raised when a parser fails to extract text from a file."""


class UnsupportedChunkingStrategyError(DocumentError):
    """Raised when no chunker is registered for a chunking strategy."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        super().__init__(f"No chunker registered for strategy '{strategy}'.")


class InvalidChunkingParametersError(DocumentError):
    """Raised when the resolved chunk_size/overlap combination is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
