"""TokenPlan API, quota monitoring, model catalog, and API key management."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import httpx

from .auth import AuthManager
from .config import InkStoneConfig

TOKENS_PER_CREDIT = 20_000_000  # Official platform conversion: 1 墨点 ≈ 20M tokens


class TokenPlanManager:
    """Manages TokenPlan quotas, balances, models, and API keys."""

    def __init__(self, config: Optional[InkStoneConfig] = None, auth: Optional[AuthManager] = None):
        self.config = config or InkStoneConfig.load()
        self.auth = auth or AuthManager(self.config)
        self.base_url = self.config.discovery_url

    def get_balance(self) -> Dict[str, Any]:
        """Fetch current credit (墨点) balance, limits, and token approximations."""
        url = f"{self.base_url}/api/tokenplan/v1/credits/balance"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"TokenPlan API error: {data.get('msg')} (code: {data.get('code')})")

        payload = data.get("data", {})
        credits_str = payload.get("available_credits", "0.0")
        try:
            credits_float = float(credits_str)
        except ValueError:
            credits_float = 0.0

        approx_tokens = int(credits_float * TOKENS_PER_CREDIT)

        return {
            "available_credits": credits_float,
            "approx_tokens": approx_tokens,
            "approx_tokens_formatted": f"{approx_tokens:,}",
            "rpm_limit": payload.get("rpm_limit", 50),
            "tpm_limit": payload.get("tpm_limit", 2_000_000),
            "usage_windows": payload.get("usage_windows", {}),
            "calculated_at": payload.get("calculated_at", ""),
            "raw": payload,
        }

    def list_models(self, page: int = 1, page_size: int = 50) -> List[Dict[str, Any]]:
        """List all available models in the TokenPlan catalog."""
        url = f"{self.base_url}/api/tokenplan/v1/models"
        headers = self.auth.get_portal_headers()
        params = {"page": page, "page_size": page_size}

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to fetch models: {data.get('msg')} (code: {data.get('code')})")

        return data.get("data", {}).get("models", [])

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all TokenPlan API keys on the account."""
        url = f"{self.base_url}/api/tokenplan/v1/keys"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to list API keys: {data.get('msg')}")

        return data.get("data", {}).get("items", [])

    def create_key(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new API key for model inference."""
        url = f"{self.base_url}/api/tokenplan/v1/keys"
        headers = self.auth.get_portal_headers()
        headers["Idempotency-Key"] = str(uuid.uuid4())

        key_name = name or f"inkstone-key-{int(uuid.uuid4().hex[:6], 16)}"
        payload = {"name": key_name[:20]}

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to create API key: {data.get('msg')} (code: {data.get('code')})")

        created = data.get("data", {})
        # If created successfully, auto-save key to config if no key exists
        if created.get("key") and not self.config.api_key:
            self.config.api_key = created["key"]
            self.config.save()

        return created

    def disable_key(self, key_id: str) -> bool:
        """Disable an API key."""
        url = f"{self.base_url}/api/tokenplan/v1/keys/{key_id}/disable"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("code") == 0

    def delete_key(self, key_id: str) -> bool:
        """Delete an API key."""
        url = f"{self.base_url}/api/tokenplan/v1/keys/{key_id}"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("code") == 0

    def get_free_grant_status(self) -> Dict[str, Any]:
        """Check status of free monthly package grant."""
        url = f"{self.base_url}/api/tokenplan/v1/users/free-grant-status"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("data", {})
