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
_CMD_GET_IMAGE = 0x62

_ACK = 0x30
_NACK = 0x31

_IMAGE_WIDTH = 258
_IMAGE_HEIGHT = 202
_IMAGE_BYTES = _IMAGE_WIDTH * _IMAGE_HEIGHT  # 52116
_DATA_PAYLOAD = 128
_DATA_PACKETS = 407  # 407 * 128 == 52116
# Per ADH data packet: 5A A5 + device(2) + packet#(2) + length(2) + data(128) + checksum(2)
_DATA_PACKET_SIZE = 138


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


def _read_one_data_packet(stream: BinaryIO, timeout: float, packet_index: int) -> bytes:
    """Read a single 138-byte data packet; return 128-byte image payload."""
    header = _read_exact(stream, 2, timeout)
    if header != bytes(_DATA_START):
        raise ValueError(
            f"Bad data packet #{packet_index}: expected 5A A5, got {header.hex()}"
        )
    _read_exact(stream, 2, timeout)  # device id
    _read_exact(stream, 4, timeout)  # packet number + valid data length
    payload = _read_exact(stream, _DATA_PAYLOAD, timeout)
    _read_exact(stream, 2, timeout)  # checksum
    return payload


def _read_image_packets(
    stream: BinaryIO, timeout: float, *, verbose: bool = True
) -> bytes:
    """Read 407 data packets (128 payload bytes each) after GetImage."""
    image = bytearray()
    progress_step = max(1, _DATA_PACKETS // 20)
    for packet_index in range(_DATA_PACKETS):
        image.extend(_read_one_data_packet(stream, timeout, packet_index))
        if verbose and (
            packet_index == 0
            or packet_index + 1 == _DATA_PACKETS
            or (packet_index + 1) % progress_step == 0
        ):
            print(
                f"\r  Downloading image: packet {packet_index + 1}/{_DATA_PACKETS} "
                f"({len(image)}/{_IMAGE_BYTES} bytes)",
                end="",
                flush=True,
            )
    if verbose:
        print(flush=True)
    return bytes(image[:_IMAGE_BYTES])


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
    image_timeout = max(timeout, 30.0)

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

            raw = _read_image_packets(ser, image_timeout, verbose=verbose)
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
