"""Runtime defaults for Raspberry Pi + GT-521Fxx sensor."""

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
DB_PATH = "fingerprint.db"
DEFAULT_USER_ID = "user_001"

# Reject stripe/corrupt captures with too few SIFT features.
MIN_KEYPOINTS = 50
MIN_MATCHES = 10
AUTH_SCORE_THRESHOLD = 8.0
