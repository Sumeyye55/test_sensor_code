import argparse
import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from cryptography.fernet import Fernet


DB_PATH = "/Users/maryamasgarova/Desktop/graduation/matching algo/fingerprint.db"
DEFAULT_USER_ID = "user_101"
DEFAULT_IMAGE_PATH = "/Users/maryamasgarova/Desktop/graduation/matching algo/data_check/same_1/101_6.tif"


def load_tif_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


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


def load_fingerprint_bgr(
    *,
    image_path: str | None = None,
    serial_port: str | None = None,
) -> np.ndarray:
    if serial_port:
        from gt521_capture import capture_bgr

        return capture_bgr(port=serial_port)
    if image_path:
        return load_tif_bgr(image_path)
    raise ValueError("Provide either image_path or serial_port.")


def register_user(
    user_id: str,
    db_path: str = DB_PATH,
    *,
    image_path: str | None = None,
    serial_port: str | None = None,
) -> tuple[int, int]:
    image = load_fingerprint_bgr(image_path=image_path, serial_port=serial_port)
    keypoints, descriptors = extract_features(image)
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
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return keypoints.shape[0], descriptors.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a user's fingerprint features into SQLite."
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"Unique user ID (default: {DEFAULT_USER_ID})",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to fingerprint image file (use instead of --port).",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for GT-521Fxx sensor (e.g. /dev/ttyUSB1 on Raspberry Pi).",
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help=f"SQLite path (default: {DB_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.port and not args.image_path:
        args.image_path = DEFAULT_IMAGE_PATH
    kp_count, desc_count = register_user(
        args.user_id,
        args.db_path,
        image_path=args.image_path,
        serial_port=args.port,
    )
    print(
        f"User '{args.user_id}' registered successfully. "
        f"Stored keypoints: {kp_count}, descriptors: {desc_count}."
    )
