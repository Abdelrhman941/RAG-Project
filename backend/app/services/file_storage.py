from pathlib import Path

import aiofiles
from fastapi import UploadFile

from ..core import EmptyFileError, FileTooLargeError

CHUNK_SIZE = 1024 * 1024  # 1 MiB


class FileStorageService:
    def __init__(self, max_size_bytes: int) -> None:
        self._max_size_bytes = max_size_bytes

    async def save_uploaded_file(
        self,
        file: UploadFile,
        destination: Path,
    ) -> int:
        """Save an uploaded file to disk in chunks to optimize memory."""
        size_bytes = 0
        try:
            async with aiofiles.open(destination, "wb") as buffer:
                while chunk := await file.read(CHUNK_SIZE):
                    size_bytes += len(chunk)
                    if size_bytes > self._max_size_bytes:
                        raise FileTooLargeError(self._max_size_bytes)
                    await buffer.write(chunk)

            if size_bytes == 0:
                raise EmptyFileError("Empty files are not allowed.")
            return size_bytes

        except Exception:
            destination.unlink(missing_ok=True)
            raise

        finally:
            await file.close()
