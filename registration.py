import argparse
import os
import sqlite3
from datetime import datetime, timezone

import cv2
import numpy as np
from cryptography.fernet import Fernet

from config import BAUD_RATE, DB_PATH, DEFAULT_USER_ID, MIN_KEYPOINTS, SERIAL_PORT


def capture_fingerprint_bgr(
    port: str = SERIAL_PORT,
    baud: int = BAUD_RATE,
    *,
    verbose: bool = True,
) -> np.ndarray:
    from gt521_capture import capture_bgr

    return capture_bgr(port=port, baud=baud, verbose=verbose)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    return clahe.apply(gray)


def extract_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preprocessed = preprocess_image(image)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(preprocessed, None)

    kp_array = np.array(
        [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in keypoints],
        dtype=np.float32,
    )
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors, dtype=np.float32)

    return kp_array, descriptors


def _array_to_blob(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array.astype(np.float32)).tobytes()


def _get_cipher() -> Fernet:
    key = os.getenv("FINGERPRINT_DB_KEY")
    if not key:
        raise ValueError(
            "Missing encryption key. Set FINGERPRINT_DB_KEY to a Fernet key."
        )
    return Fernet(key.encode("utf-8"))


def _encrypt_blob(blob: bytes, cipher: Fernet) -> bytes:
    return cipher.encrypt(blob)


def register_user(
    user_id: str,
    db_path: str = DB_PATH,
    *,
    port: str = SERIAL_PORT,
    baud: int = BAUD_RATE,
    verbose: bool = True,
    save_capture_path: str | None = None,
) -> tuple[int, int]:
    image = capture_fingerprint_bgr(port=port, baud=baud, verbose=verbose)
    if save_capture_path:
        cv2.imwrite(save_capture_path, image)
        if verbose:
            print(f"Saved sensor capture to {save_capture_path}", flush=True)
    if verbose:
        print("Extracting SIFT features...", flush=True)
    keypoints, descriptors = extract_features(image)
    if keypoints.shape[0] < MIN_KEYPOINTS or descriptors.shape[0] < MIN_KEYPOINTS:
        raise ValueError(
            f"Only {keypoints.shape[0]} keypoints detected (need >={MIN_KEYPOINTS}). "
            "Sensor image looks invalid — fix gt521_capture.py, verify with "
            "python3 gt521_capture.py --output test.png, then register again."
        )
    if verbose:
        print(f"Saving encrypted features for '{user_id}'...", flush=True)
    cipher = _get_cipher()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_fingerprints (user_id, keypoints, descriptors, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                _encrypt_blob(_array_to_blob(keypoints), cipher),
                _encrypt_blob(_array_to_blob(descriptors), cipher),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return keypoints.shape[0], descriptors.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a user from a live GT-521Fxx sensor capture."
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument(
        "--save-capture",
        default=None,
        help="Optional PNG path to save the exact image sent to SIFT (for debugging).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kp_count, desc_count = register_user(
        args.user_id,
        args.db_path,
        port=args.port,
        baud=args.baud,
        save_capture_path=args.save_capture,
    )
    print(
        f"User '{args.user_id}' registered from sensor. "
        f"Stored keypoints: {kp_count}, descriptors: {desc_count}."
    )
