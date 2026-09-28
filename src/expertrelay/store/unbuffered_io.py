"""Unbuffered (OS-cache-bypassing) file I/O primitives.

Two users: the expert store reader, which loads one expert with one read
straight from the SSD, and the machine profiler, which times such reads.
They share this module so there is only one implementation of the
alignment rules and the Win32 plumbing.

Why unbuffered: an expert load must cost what the SSD costs, not what the
Windows file cache happens to hold. It also must not push other data out
of an 8 GB machine's RAM by filling the cache with 12 GB of experts.
FILE_FLAG_NO_BUFFERING gives both. In exchange, every read/write size, file
offset and buffer address must be a multiple of the volume sector size.
ALIGNMENT (4096) covers both 512- and 4096-byte sectors. Buffers come from
an anonymous mmap, which is page-aligned.

Windows only. Other platforms raise NotImplementedError (Linux would use
O_DIRECT; not implemented yet -- see docs/limitations.md).
"""

from __future__ import annotations

import ctypes
import mmap
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ALIGNMENT = 4096


def align_up(n: int, alignment: int = ALIGNMENT) -> int:
    return -(-n // alignment) * alignment


def is_aligned(n: int, alignment: int = ALIGNMENT) -> bool:
    return n % alignment == 0


if sys.platform == "win32":
    from ctypes import wintypes

    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _CREATE_ALWAYS = 2
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _FILE_FLAG_WRITE_THROUGH = 0x80000000
    _FILE_BEGIN = 0
    _INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.WriteFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _kernel32.WriteFile.restype = wintypes.BOOL
    _kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    _kernel32.SetFilePointerEx.restype = wintypes.BOOL
    _kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    _kernel32.SetEndOfFile.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def _require_windows() -> None:
    if sys.platform != "win32":
        raise NotImplementedError("unbuffered I/O is implemented for Windows only (see docs/limitations.md)")


def _raise_last_win_error(what: str) -> None:
    raise ctypes.WinError(ctypes.get_last_error(), f"{what} failed")


@contextmanager
def aligned_buffer(size: int) -> Iterator[tuple[mmap.mmap, int]]:
    """A page-aligned, writable buffer and its address. The ctypes view has to
    be released before the mmap can close, hence the explicit ordering."""
    mm = mmap.mmap(-1, size)
    view = (ctypes.c_char * size).from_buffer(mm)
    try:
        yield mm, ctypes.addressof(view)
    finally:
        del view
        mm.close()


@contextmanager
def unbuffered_handle(path: Path, *, write: bool) -> Iterator[int]:
    """Open `path` bypassing the OS file cache. Writing creates/truncates the
    file and also sets WRITE_THROUGH, so written data never sits in the cache."""
    _require_windows()
    if write:
        access, disposition = _GENERIC_WRITE, _CREATE_ALWAYS
        flags = _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_WRITE_THROUGH
    else:
        access, disposition = _GENERIC_READ, _OPEN_EXISTING
        flags = _FILE_FLAG_NO_BUFFERING
    handle = _kernel32.CreateFileW(str(path), access, _FILE_SHARE_READ, None, disposition, flags, None)
    if handle == _INVALID_HANDLE_VALUE:
        _raise_last_win_error(f"CreateFileW({path})")
    try:
        yield handle
    finally:
        _kernel32.CloseHandle(handle)


def read_at(handle: int, address: int, offset: int, nbytes: int) -> None:
    """One ReadFile call: `nbytes` from file `offset` into the buffer at `address`.

    Raises on any error or short read. A short read on an unbuffered handle
    means the requested range runs past end-of-file.
    """
    if not (is_aligned(offset) and is_aligned(nbytes) and is_aligned(address)):
        raise ValueError(f"unaligned unbuffered read: offset={offset} nbytes={nbytes} address={address:#x}")
    got = wintypes.DWORD(0)
    if not _kernel32.SetFilePointerEx(handle, offset, None, _FILE_BEGIN):
        _raise_last_win_error("SetFilePointerEx")
    if not _kernel32.ReadFile(handle, address, nbytes, ctypes.byref(got), None):
        _raise_last_win_error("ReadFile")
    if got.value != nbytes:
        raise OSError(f"short read at offset {offset}: {got.value} of {nbytes} bytes")


def set_size_and_rewind(handle: int, nbytes: int) -> None:
    """Give a new file its final size up front, then go back to the start.
    NTFS allocates the whole size at once, so the file is as contiguous as
    the volume's free space allows, instead of growing piece by piece.
    Sequential writes from offset 0 then never trigger zero-filling."""
    if not is_aligned(nbytes):
        raise ValueError(f"unaligned file size {nbytes}")
    if not _kernel32.SetFilePointerEx(handle, nbytes, None, _FILE_BEGIN):
        _raise_last_win_error("SetFilePointerEx")
    if not _kernel32.SetEndOfFile(handle):
        _raise_last_win_error("SetEndOfFile")
    if not _kernel32.SetFilePointerEx(handle, 0, None, _FILE_BEGIN):
        _raise_last_win_error("SetFilePointerEx")


def write_sequential(handle: int, address: int, nbytes: int) -> None:
    """One WriteFile call at the handle's current position."""
    written = wintypes.DWORD(0)
    if (
        not _kernel32.WriteFile(handle, address, nbytes, ctypes.byref(written), None)
        or written.value != nbytes
    ):
        _raise_last_win_error("WriteFile")
