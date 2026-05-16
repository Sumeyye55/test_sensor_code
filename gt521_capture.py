"""Capture grayscale fingerprint images from GT-521Fxx sensors over UART."""

from __future__ import annotations

import struct
import time
from typing import BinaryIO

import numpy as np

try:
    import serial
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyserial is required for sensor capture. Install with: pip install pyserial"
    ) from exc

# Protocol constants (GT-511C3 / GT-521Fxx family)
_CMD_START = (0x55, 0xAA)
_RSP_START = (0x55, 0xAA)
_DATA_START = (0x5A, 0xA5)
_DEVICE_ID = 0x0001

_CMD_OPEN = 0x01
_CMD_CLOSE = 0x02
_CMD_CMOS_LED = 0x12
_CMD_IS_PRESS = 0x26
_CMD_CAPTURE = 0x60
_CMD_GET_IMAGE = 0x63

_ACK = 0x30
_NACK = 0x31

_IMAGE_WIDTH = 160
_IMAGE_HEIGHT = 120
_IMAGE_BYTES = _IMAGE_WIDTH * _IMAGE_HEIGHT  # 258 * 202 = 52116
_DATA_PAYLOAD = 128  # usual payload per UART chunk
_DATA_PACKETS = 407  # ~407 chunks (datasheet); 407*128=52096, image needs +20 bytes
_IMAGE_EXTRA = _IMAGE_BYTES - (_DATA_PACKETS * _DATA_PAYLOAD)  # 20
# Framed wire sizes on the UART line (header/checksum + 128-byte payload, or 148 on 1st)
_DATA_FRAMED_SIZES = (134, 138, 148, 168)
# Observed on GT-521F52 @ 9600: ~52122 wire bytes = 6-byte header + 52116 image
_WIRE_BYTES_TARGET = _IMAGE_BYTES + 6  # 52122
# Upper bound if the module sends fully framed 407 packets (407 × 134)
_WIRE_BYTES_MAX = _DATA_PACKETS * 134  # 54538


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _checksum(data: bytes) -> int:
    return sum(data) & 0xFFFF


def _build_command(command: int, parameter: int = 0) -> bytes:
    packet = bytearray(12)
    packet[0], packet[1] = _CMD_START
    struct.pack_into("<H", packet, 2, _DEVICE_ID)
    struct.pack_into("<I", packet, 4, parameter & 0xFFFFFFFF)
    struct.pack_into("<H", packet, 8, command & 0xFFFF)
    chksum = _checksum(packet[:10])
    struct.pack_into("<H", packet, 10, chksum)
    return bytes(packet)


def _read_exact(stream: BinaryIO, size: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {size} bytes from sensor.")
        chunk = stream.read(remaining)
        if chunk:
            chunks.append(chunk)
            remaining -= len(chunk)
        else:
            time.sleep(0.01)
    return b"".join(chunks)


def _read_response(stream: BinaryIO, timeout: float) -> tuple[bool, int]:
    """Read a 12-byte response packet; return (ack_ok, parameter)."""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for sensor response.")
        first = stream.read(1)
        if not first:
            time.sleep(0.01)
            continue
        if first[0] != _RSP_START[0]:
            continue
        second = _read_exact(stream, 1, timeout)
        if second[0] != _RSP_START[1]:
            continue
        body = _read_exact(stream, 10, timeout)
        packet = bytes([_RSP_START[0], _RSP_START[1]]) + body
        expected = _checksum(packet[:10])
        received = struct.unpack_from("<H", packet, 10)[0]
        if expected != received:
            raise ValueError("Sensor response checksum mismatch.")
        response_code = struct.unpack_from("<H", packet, 8)[0]
        parameter = struct.unpack_from("<I", packet, 4)[0]
        if response_code == _ACK:
            return True, parameter
        if response_code == _NACK:
            return False, parameter
        raise ValueError(f"Unexpected response code: 0x{response_code:04X}")


def _read_image_stream(stream: BinaryIO, timeout: float, *, verbose: bool = True) -> bytes:
    """Read the full UART image transfer, then parse into 52116 image bytes."""
    _log(
        f"  Receiving image stream — hold finger still "
        f"(~{_WIRE_BYTES_TARGET} bytes, up to {timeout:.0f}s at 9600 baud)...",
        verbose=verbose,
    )
    buf = bytearray()
    deadline = time.monotonic() + timeout
    last_reported = 0

    while time.monotonic() < deadline:
        waiting = getattr(stream, "in_waiting", 0) or 0
        if waiting:
            buf.extend(stream.read(waiting))
        elif len(buf) >= _WIRE_BYTES_TARGET:
            time.sleep(0.25)
            extra = getattr(stream, "in_waiting", 0) or 0
            if extra:
                buf.extend(stream.read(extra))
            else:
                break
        else:
            time.sleep(0.05)

        if verbose and len(buf) - last_reported >= 2500:
            pct = min(100, (100 * len(buf)) // _WIRE_BYTES_TARGET)
            print(
                f"\r  Downloading image: {len(buf)}/{_WIRE_BYTES_TARGET} bytes ({pct}%)...",
                end="",
                flush=True,
            )
            last_reported = len(buf)

    if verbose:
        print(
            f"\r  Downloading image: {len(buf)}/{_WIRE_BYTES_TARGET} bytes ({min(100, (100 * len(buf)) // _WIRE_BYTES_TARGET)}%).          ",
            flush=True,
        )
        print(flush=True)

    if len(buf) < _WIRE_BYTES_TARGET:
        raise ValueError(
            f"Incomplete image download: received {len(buf)} bytes, "
            f"expected at least {_WIRE_BYTES_TARGET} "
            f"(image {_IMAGE_BYTES} + header). "
            "Keep your finger on the sensor until the progress shows 100%."
        )

    return _parse_image_buffer(bytes(buf))


def _try_framed_all(buf: bytes, start: int, frame_size: int, payload_skip: int) -> bytes | None:
    """All 407 chunks are framed with the same wire size."""
    image = bytearray()
    for index in range(_DATA_PACKETS):
        base = start + index * frame_size
        if base + payload_skip + _DATA_PAYLOAD > len(buf):
            return None
        if buf[base] != _DATA_START[0] or buf[base + 1] != _DATA_START[1]:
            return None
        image.extend(buf[base + payload_skip : base + payload_skip + _DATA_PAYLOAD])
    return bytes(image) if len(image) == _IMAGE_BYTES else None


def _try_first_framed_then_raw(buf: bytes, start: int, first_size: int, skip: int) -> bytes | None:
    """First chunk has a header; remaining 406 chunks are raw 128-byte payloads."""
    if start + skip + _DATA_PAYLOAD > len(buf):
        return None
    image = bytearray(buf[start + skip : start + skip + _DATA_PAYLOAD])
    pos = start + first_size
    for _ in range(_DATA_PACKETS - 1):
        if pos + _DATA_PAYLOAD > len(buf):
            return None
        image.extend(buf[pos : pos + _DATA_PAYLOAD])
        pos += _DATA_PAYLOAD
    return bytes(image) if len(image) == _IMAGE_BYTES else None


def _try_first_148_then_raw_128(buf: bytes, start: int, first_wire_size: int, skip: int) -> bytes | None:
    """First payload is 148 bytes, then 406 x 128 bytes (148 + 406*128 = 52116)."""
    first_payload = _DATA_PAYLOAD + _IMAGE_EXTRA  # 148
    if start + skip + first_payload > len(buf):
        return None
    image = bytearray(buf[start + skip : start + skip + first_payload])
    pos = start + first_wire_size
    for _ in range(_DATA_PACKETS - 1):
        if pos + _DATA_PAYLOAD > len(buf):
            return None
        image.extend(buf[pos : pos + _DATA_PAYLOAD])
        pos += _DATA_PAYLOAD
    return bytes(image) if len(image) == _IMAGE_BYTES else None


def _extract_sequential_payloads(buf: bytes, start: int) -> bytes | None:
    """Concatenate 407 x 128-byte payloads, skipping headers/checksums between chunks."""
    for header_skip in (4, 6, 8, 12, 20, 26):
        image = bytearray()
        pos = start + header_skip
        if pos + 2 <= len(buf) and buf[pos] == _DATA_START[0] and buf[pos + 1] == _DATA_START[1]:
            pos += 4

        ok = True
        for _ in range(_DATA_PACKETS):
            if len(image) >= _IMAGE_BYTES:
                break
            if pos + 2 <= len(buf) and buf[pos] == _DATA_START[0] and buf[pos + 1] == _DATA_START[1]:
                pos += 4
            if pos + _DATA_PAYLOAD > len(buf):
                ok = False
                break
            image.extend(buf[pos : pos + _DATA_PAYLOAD])
            pos += _DATA_PAYLOAD
            if pos + 2 <= len(buf) and buf[pos] != _DATA_START[0]:
                pos += 2

        if ok and len(image) == _IMAGE_BYTES:
            return bytes(image)
    return None


def _try_compact_image_block(buf: bytes, start: int) -> bytes | None:
    """Last resort: one contiguous block (often produces stripe artifacts if wrong)."""
    for skip in (4, 6, 8, 12, 20):
        end = start + skip + _IMAGE_BYTES
        if end <= len(buf):
            return bytes(buf[start + skip : end])
    return None


def _parse_image_buffer(buf: bytes) -> bytes:
    """Extract 52116 image bytes (258x202) from the sensor download stream."""
    start = buf.find(bytes(_DATA_START))
    if start < 0:
        head = buf[:20].hex() if buf else "empty"
        raise ValueError(
            f"Image stream missing 5A A5 marker. Received {len(buf)} bytes, head={head}"
        )

    sequential = _extract_sequential_payloads(buf, start)
    if sequential is not None:
        return sequential

    for frame_size in _DATA_FRAMED_SIZES:
        for payload_skip in (4, 8, 12, 20):
            if payload_skip + _DATA_PAYLOAD > frame_size:
                continue
            result = _try_framed_all(buf, start, frame_size, payload_skip)
            if result is not None:
                return result

    for first_size in _DATA_FRAMED_SIZES:
        for skip in (4, 8, 12, 20):
            result = _try_first_framed_then_raw(buf, start, first_size, skip)
            if result is not None:
                return result
            result = _try_first_148_then_raw_128(buf, start, first_size, skip)
            if result is not None:
                return result

    compact = _try_compact_image_block(buf, start)
    if compact is not None:
        return compact

    raise ValueError(
        f"Could not parse image from {len(buf)}-byte stream "
        f"(start index {start}). Try capture again with finger held still."
    )


def _send_command(
    stream: serial.Serial,
    command: int,
    parameter: int = 0,
    *,
    flush_input: bool = True,
) -> tuple[bool, int]:
    if flush_input:
        stream.reset_input_buffer()
    stream.write(_build_command(command, parameter))
    return _read_response(stream, timeout=stream.timeout or 2.0)


def _safe_command(stream: serial.Serial, command: int, parameter: int = 0) -> None:
    try:
        _send_command(stream, command, parameter)
    except (TimeoutError, ValueError, OSError):
        pass


def capture_grayscale(
    port: str = "/dev/ttyUSB1",
    baud: int = 9600,
    timeout: float = 10.0,
    wait_finger: bool = True,
    finger_wait_seconds: float = 30.0,
    high_quality: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Capture a 258x202 grayscale image from a GT-521Fxx sensor.

    Typical use on Raspberry Pi: port="/dev/ttyUSB1".
    """
    # 9600 baud + ~55 KB image data needs well over 30s on the wire.
    image_timeout = max(timeout, 90.0)

    _log(f"Connecting to sensor on {port} @ {baud} baud...", verbose=verbose)

    with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
        ok, _ = _send_command(ser, _CMD_OPEN)
        if not ok:
            raise RuntimeError("Sensor did not ACK Open command.")
        _log("Sensor ready.", verbose=verbose)

        try:
            _send_command(ser, _CMD_CMOS_LED, 1)
            _log("LED on.", verbose=verbose)

            if wait_finger:
                _log(
                    f"\n>>> PRESS YOUR FINGER on the sensor now "
                    f"(waiting up to {finger_wait_seconds:.0f}s) <<<",
                    verbose=verbose,
                )
                deadline = time.monotonic() + finger_wait_seconds
                last_hint = 0.0
                while time.monotonic() < deadline:
                    ok, pressed = _send_command(ser, _CMD_IS_PRESS)
                    if ok and pressed == 0:
                        _log(
                            "Finger detected — hold still, do not lift yet.",
                            verbose=verbose,
                        )
                        break
                    now = time.monotonic()
                    if verbose and now - last_hint >= 2.0:
                        remaining = int(deadline - now)
                        print(
                            f"  Still waiting for finger... ({remaining}s left)",
                            flush=True,
                        )
                        last_hint = now
                    time.sleep(0.15)
                else:
                    raise TimeoutError(
                        "No finger detected on sensor within "
                        f"{finger_wait_seconds:.0f}s."
                    )

            _log("Capturing fingerprint scan...", verbose=verbose)
            ok, _ = _send_command(ser, _CMD_CAPTURE, 1 if high_quality else 0)
            if not ok:
                raise RuntimeError(
                    "CaptureFinger failed. Place finger firmly on the sensor."
                )
            _log("Scan captured.", verbose=verbose)

            _log("Starting image download from sensor...", verbose=verbose)
            ok, _ = _send_command(ser, _CMD_GET_IMAGE)
            if not ok:
                raise RuntimeError("GetImage command was rejected by the sensor.")

            raw = _read_image_stream(ser, _IMAGE_BYTES, image_timeout)
            _log(
                f"Image download complete ({len(raw)} bytes, "
                f"{_IMAGE_WIDTH}x{_IMAGE_HEIGHT}).",
                verbose=verbose,
            )
        finally:
            _log("Turning LED off and closing sensor...", verbose=verbose)
            _safe_command(ser, _CMD_CMOS_LED, 0)
            _safe_command(ser, _CMD_CLOSE)

    _log("Done. You can lift your finger.", verbose=verbose)
    return np.frombuffer(raw, dtype=np.uint8).reshape(_IMAGE_HEIGHT, _IMAGE_WIDTH)


def capture_bgr(
    port: str = "/dev/ttyUSB1",
    baud: int = 9600,
    timeout: float = 10.0,
    **kwargs,
) -> np.ndarray:
    """Return captured image as BGR (OpenCV-style) for extract_features()."""
    import cv2

    gray = capture_grayscale(port=port, baud=baud, timeout=timeout, **kwargs)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    import argparse

    import cv2

    parser = argparse.ArgumentParser(
        description="Capture a fingerprint image from GT-521Fxx and save as PNG."
    )
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--output", default="fingerprint_capture.png")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress messages on the terminal.",
    )
    args = parser.parse_args()

    gray = capture_grayscale(
        port=args.port,
        baud=args.baud,
        wait_finger=not args.no_wait,
        verbose=not args.quiet,
    )
    cv2.imwrite(args.output, gray)
    print(f"Saved {args.output} ({gray.shape[1]}x{gray.shape[0]})")
