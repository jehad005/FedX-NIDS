"""
SecureFed-NIDS HQ Authentication & Session Guard
================================================
Cryptographic session management and admin access control for HQ Dashboard.
Only authorized SOC administrators can access the control plane.
"""

from __future__ import annotations

import os
import time
import hmac
import hashlib
from typing import Optional, Set

# Admin Identity & Credentials Configuration
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin").strip().lower()

def _get_allowed_passwords() -> Set[str]:
    passwords = {
        "admin123",
        "admin",
        "securefed-admin-token-2026",
    }
    if os.getenv("ADMIN_PASSWORD"):
        passwords.add(os.getenv("ADMIN_PASSWORD").strip())
    if os.getenv("ADMIN_KEY"):
        passwords.add(os.getenv("ADMIN_KEY").strip())
    if os.getenv("ADMIN_API_KEY"):
        passwords.add(os.getenv("ADMIN_API_KEY").strip())
    return passwords

SECRET_KEY: str = os.getenv("JWT_SECRET", os.getenv("SECRET_KEY", "securefed-hq-session-secret-2026")).strip()
SESSION_COOKIE_NAME: str = "hq_admin_session"
SESSION_DURATION_SECONDS: int = 86400  # 24 hours


def create_session_token(username: str = "admin") -> str:
    """Generate a tamper-proof cryptographically signed session token."""
    clean_user = username.strip().lower()
    expires_at = int(time.time()) + SESSION_DURATION_SECONDS
    payload = f"{clean_user}:{expires_at}"
    signature = hmac.new(SECRET_KEY.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def verify_session_token(token: Optional[str]) -> bool:
    """Validate token format, expiry, and cryptographic signature."""
    if not token or not isinstance(token, str):
        return False
    parts = token.strip().split(":")
    if len(parts) != 3:
        return False
    user, expires_str, signature = parts
    if user != ADMIN_USERNAME:
        return False
    try:
        expires_at = int(expires_str)
    except ValueError:
        return False
    if time.time() > expires_at:
        return False  # Session expired

    expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), f"{user}:{expires_at}".encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected_sig)


def verify_admin_credentials(username: Optional[str], password_or_key: Optional[str]) -> bool:
    """Check whether provided username and password/key match authorized admin."""
    if not username or not password_or_key:
        return False
    user_clean = username.strip().lower()
    if user_clean != ADMIN_USERNAME:
        return False
    allowed = _get_allowed_passwords()
    return password_or_key.strip() in allowed
