import argparse
import os
import sqlite3

import cv2
import numpy as np
from cryptography.fernet import Fernet

from registration import extract_features, load_fingerprint_bgr


DB_PATH = "/Users/maryamasgarova/Desktop/graduation/matching algo/fingerprint.db"
DEFAULT_USER_ID = "user_101"
DEFAULT_IMAGE_PATH = "/Users/maryamasgarova/Desktop/graduation/matching algo/data_check/different_7/103_1.tif"
# Match score is len(matches)/min(keypoints)*100 (same as matching.ipynb). >= 1: accept, < 1: reject.
AUTH_SCORE_THRESHOLD = 1.0


def _get_cipher() -> Fernet:
    key = os.getenv("FINGERPRINT_DB_KEY")
    if not key:
        raise ValueError(
            "Missing encryption key. Set FINGERPRINT_DB_KEY to a Fernet key."
        )
    return Fernet(key.encode("utf-8"))


def _decrypt_blob(blob: bytes, cipher: Fernet) -> bytes:
    return cipher.decrypt(blob)


def _blob_to_array(blob: bytes, cols: int) -> np.ndarray:
    if not blob:
        return np.empty((0, cols), dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32).reshape(-1, cols)


def fetch_user_features(
    user_id: str, db_path: str
) -> tuple[np.ndarray, np.ndarray] | None:
    cipher = _get_cipher()
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT keypoints, descriptors FROM user_fingerprints WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    kp_blob_enc, desc_blob_enc = row
    kp_blob = _decrypt_blob(kp_blob_enc, cipher)
    desc_blob = _decrypt_blob(desc_blob_enc, cipher)
    keypoints = _blob_to_array(kp_blob, cols=4)
    descriptors = _blob_to_array(desc_blob, cols=128)
    return keypoints, descriptors


def compute_match_score(
    probe_keypoints: np.ndarray,
    probe_descriptors: np.ndarray,
    db_keypoints: np.ndarray,
    db_descriptors: np.ndarray,
) -> float:
    """Same pipeline as matching.ipynb (cells 4–8).

    Notebook: image 1 = sample (enrolled), image 2 = check (probe).
    flann.knnMatch(descriptors_1, descriptors_2) → queryIdx indexes image 1,
    trainIdx indexes image 2. Here descriptors_1 = DB, descriptors_2 = probe.
    """
    if probe_descriptors.size == 0 or db_descriptors.size == 0:
        return 0.0

    db_d = np.ascontiguousarray(db_descriptors, dtype=np.float32)
    probe_d = np.ascontiguousarray(probe_descriptors, dtype=np.float32)

    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {})
    matches = flann.knnMatch(db_d, probe_d, k=2)
    match_points = []
    for pair in matches:
        if len(pair) < 2:
            continue
        p, q = pair[0], pair[1]
        if p.distance < 0.7 * q.distance:
            match_points.append(p)

    if len(match_points) > 4:
        src_pts = np.float32(
            [db_keypoints[m.queryIdx][:2] for m in match_points]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [probe_keypoints[m.trainIdx][:2] for m in match_points]
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            match_points = [m for m, v in zip(match_points, mask.ravel()) if v == 1]

    keypoints_count = min(len(probe_keypoints), len(db_keypoints))
    if keypoints_count == 0:
        return 0.0

    return len(match_points) / keypoints_count


def authenticate(
    user_id: str,
    db_path: str,
    *,
    image_path: str | None = None,
    serial_port: str | None = None,
) -> tuple[float | None, bool]:
    probe_image = load_fingerprint_bgr(image_path=image_path, serial_port=serial_port)
    probe_keypoints, probe_descriptors = extract_features(probe_image)
    stored = fetch_user_features(user_id, db_path)

    if stored is None:
        return None, False

    db_keypoints, db_descriptors = stored
    ratio = compute_match_score(
        probe_keypoints, probe_descriptors, db_keypoints, db_descriptors
    )
    scaled_score = ratio * 100.0
    ok = scaled_score >= AUTH_SCORE_THRESHOLD
    return scaled_score, ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate a fingerprint image against encrypted DB features."
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"User ID to authenticate against. Default: {DEFAULT_USER_ID}",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to input fingerprint image (use instead of --port).",
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
    score, ok = authenticate(
        args.user_id,
        args.db_path,
        image_path=args.image_path,
        serial_port=args.port,
    )
    if score is None:
        print("fail")
    else:
        print(score)
        print("success" if ok else "fail")
