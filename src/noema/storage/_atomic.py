"""Crash-safe file writes for the storage layer.

Writing a bundle or triple-store file directly to its final path means a
crash, a full disk, or an exception partway through leaves a truncated file
*in place of* the previous good one. `atomic_write` writes to a sibling
temporary file instead and renames it over the destination only after the
write succeeded and was flushed to disk, so a reader (or a later load after a
crash) sees either the complete old file or the complete new one.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator
from typing import IO


@contextlib.contextmanager
def atomic_write(path: str | os.PathLike[str]) -> Iterator[IO[bytes]]:
    """Yield a binary file object; on clean exit, atomically replace `path`.

    The temporary file lives in the same directory as `path` (a rename is
    only atomic within one filesystem) and is removed if anything inside the
    `with` block raises, leaving any existing file at `path` untouched.
    """
    target = os.fspath(path)
    tmp = f"{target}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "xb") as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise