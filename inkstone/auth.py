"""Authentication, token introspection, and header generation for Intern InkStone."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Optional

from .config import InkStoneConfig


class AuthManager:
    """Manages session tokens, JWT validity, and authentication headers."""

    def __init__(self, config: Optional[InkStoneConfig] = None):
        self.config = config or InkStoneConfig.load()

    def decode_jwt_payload(self) -> Dict[str, Any]:
        """Decode the payload of the configured uaa_token without external dependencies."""
        token = self.config.uaa_token
        if not token or "." not in token:
            return {}
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return {}
            payload_b64 = parts[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            return json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8"))
        except Exception:
            return {}

    def get_token_status(self) -> Dict[str, Any]:
        """Return token expiry and role status."""
        payload = self.decode_jwt_payload()
        if not payload:
            return {"valid": False, "reason": "No valid JWT configured"}

        exp = payload.get("exp", 0)
        now = time.time()
        remaining_seconds = max(0, int(exp - now))
        is_expired = remaining_seconds <= 0

        exp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(exp)) if exp else "Unknown"

        return {
            "valid": not is_expired,
            "expired": is_expired,
            "sso_uid": payload.get("jti", self.config.ssouid),
            "role": payload.get("rol", "ROLE_REGISTER"),
            "issuer": payload.get("iss", "OpenXLab"),
            "expires_at_timestamp": exp,
            "expires_at_readable": exp_str,
            "remaining_seconds": remaining_seconds,
            "remaining_hours": round(remaining_seconds / 3600, 2),
            "remaining_days": round(remaining_seconds / 86400, 2),
        }

    def get_portal_headers(self) -> Dict[str, str]:
        """Generate headers required for the discovery portal and backend management APIs."""
        token = self.config.uaa_token
        uid = self.config.ssouid
        acw_tc = self.config.acw_tc

        cookie_parts = []
        if uid:
            cookie_parts.append(f"ssouid={uid}")
        if token:
            cookie_parts.append(f"uaa-token={token}")
            cookie_parts.append(f"token={token}")
        if acw_tc:
            cookie_parts.append(f"acw_tc={acw_tc}")

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Referer": f"{self.config.discovery_url}/",
            "Origin": self.config.discovery_url,
        }

        if cookie_parts:
            headers["Cookie"] = "; ".join(cookie_parts)
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["token"] = token
            headers["uaa-token"] = token

        return headers

    def get_inference_headers(self, protocol: str = "openai") -> Dict[str, str]:
        """Generate headers for model inference endpoints."""
        api_key = self.config.api_key
        headers = {
            "User-Agent": "InkStone-Automation/1.0",
            "Content-Type": "application/json",
        }
        if api_key:
            if protocol == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        return headers
