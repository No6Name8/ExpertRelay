"""Minimal length-prefixed request/response framing over a plain TCP socket.

This is the actual wire protocol used between the coordinator ("Process A")
and an expert host ("Process B"). It doesn't know or care whether the peer
is on 127.0.0.1 or a machine across the room -- that's the whole point: the
only thing that changes when Process B moves to a real second machine is
the --host argument each side is started with.

Wire format per message:
    [4 bytes] big-endian u32: length of JSON header, in bytes
    [N bytes] UTF-8 JSON header (small metadata: cmd, layer, expert_id, shape, dtype)
    [4 bytes] big-endian u32: length of raw payload, in bytes
    [M bytes] raw payload (a float32 tensor's bytes, or empty)
"""
from __future__ import annotations

import json
import struct
import socket
from typing import Tuple

import numpy as np

_LEN = struct.Struct(">I")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(_LEN.pack(len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(_LEN.pack(len(payload)))
    if payload:
        sock.sendall(payload)


def recv_msg(sock: socket.socket) -> Tuple[dict, bytes]:
    (hlen,) = _LEN.unpack(recv_exact(sock, 4))
    header = json.loads(recv_exact(sock, hlen))
    (plen,) = _LEN.unpack(recv_exact(sock, 4))
    payload = recv_exact(sock, plen) if plen else b""
    return header, payload


def array_to_payload(arr: np.ndarray) -> Tuple[dict, bytes]:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return {"shape": list(arr.shape), "dtype": "float32"}, arr.tobytes()


def payload_to_array(header: dict, payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.dtype(header["dtype"])).reshape(header["shape"]).copy()
