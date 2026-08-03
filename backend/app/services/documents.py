from pathlib import Path

import aiofiles
from fastapi import UploadFile

from ..core import EmptyFileError, FileTooLargeError

CHUNK_SIZE = 1024 * 1024  # 1 MiB


async def save_local_file(
    file: UploadFile,
    destination: Path,
    max_size_bytes: int,
) -> int:
    size_bytes = 0
    try:
        async with aiofiles.open(destination, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_size_bytes:
                    raise FileTooLargeError(max_size_bytes)
                await buffer.write(chunk)
        if size_bytes == 0:
            raise EmptyFileError("Empty files are not allowed.")
        return size_bytes

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await file.close()
