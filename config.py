"""Runtime defaults for Raspberry Pi + GT-521Fxx sensor."""

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
DB_PATH = "fingerprint.db"
DEFAULT_USER_ID = "user_001"

# Reject stripe/corrupt captures with too few SIFT features.
MIN_KEYPOINTS = 50
MIN_MATCHES = 8
# Notebook .tif scans ~8+; GT-521 live captures are often 4–7 for the same finger.
# Raise if strangers still pass; lower if your own finger fails (try 4.0–6.0).
AUTH_SCORE_THRESHOLD = 5.0
