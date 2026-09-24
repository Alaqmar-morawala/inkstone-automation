"""Unified InkStone client facade with multi-account support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .auth import AuthManager
from .compute import ComputeManager
from .config import InkStoneConfig
from .deploy import ModelDeployer
from .images import ImageManager
from .inference import InferenceClient
from .tokenplan import TokenPlanManager


class InkStoneClient:
    """Consolidated client integrating TokenPlan, Model Inference, Dev Machine Compute, and Custom Images."""

    def __init__(self, config_path: Optional[str | Path] = None, account: Optional[str] = None):
        self.config_path = config_path
        self.config = InkStoneConfig.load(config_path, account=account)
        self._reinit_subsystems()
        self.deployer = ModelDeployer()

    def _reinit_subsystems(self) -> None:
        self.auth = AuthManager(self.config)
        self.tokenplan = TokenPlanManager(self.config, self.auth)
        self.inference = InferenceClient(self.config, self.auth)
        self.compute = ComputeManager(self.config, self.auth)
        self.images = ImageManager(self.config, self.auth)

    def switch_account(self, account_name: str) -> None:
        """Switch active account profile."""
        self.config.switch_account(account_name)
        self._reinit_subsystems()

    def get_full_status(self) -> Dict[str, Any]:
        """Fetch consolidated authentication, token balance, and compute quota status for active account."""
        auth_status = self.auth.get_token_status()
        token_balance = {}
        compute_balance = {}

        try:
            token_balance = self.tokenplan.get_balance()
        except Exception as e:
            token_balance = {"error": str(e)}

        try:
            compute_balance = self.compute.get_user_resource()
        except Exception as e:
            compute_balance = {"error": str(e)}

        acc_name = self.config.active_account
        acc_info = self.config.accounts.get(acc_name, {})

        return {
            "account": {
                "profile_name": acc_name,
                "username": acc_info.get("username", "Unknown"),
                "sso_uid": auth_status.get("sso_uid", self.config.ssouid),
                "token_valid": auth_status.get("valid", False),
                "expires_at": auth_status.get("expires_at_readable", "Unknown"),
                "remaining_days": auth_status.get("remaining_days", 0),
            },
            "tokens": token_balance,
            "compute": compute_balance,
        }

    def get_pool_status(self) -> Dict[str, Any]:
        """Fetch statuses for ALL configured accounts and calculate pooled totals."""
        current_acc = self.config.active_account
        accounts_detail = {}

        total_credits = 0.0
        total_tokens = 0
        total_compute_points = 0
        total_a100_hours = 0.0
        total_ascend_hours = 0.0

        for acc_name in self.config.accounts:
            self.switch_account(acc_name)
            st = self.get_full_status()
            accounts_detail[acc_name] = st

            tok = st.get("tokens", {})
            if "available_credits" in tok:
                total_credits += tok["available_credits"]
                total_tokens += tok.get("approx_tokens", 0)

            cmp = st.get("compute", {})
            if "remaining_points" in cmp:
                total_compute_points += cmp["remaining_points"]
                total_a100_hours += cmp.get("a100_hours_remaining", 0.0)
                total_ascend_hours += cmp.get("ascend_hours_remaining", 0.0)

        # Restore original account
        self.switch_account(current_acc)

        return {
            "accounts_count": len(self.config.accounts),
            "accounts": accounts_detail,
            "pooled": {
                "total_credits": round(total_credits, 6),
                "total_tokens": total_tokens,
                "total_tokens_formatted": f"{total_tokens:,}",
                "total_compute_points": total_compute_points,
                "total_a100_hours": round(total_a100_hours, 1),
                "total_ascend_hours": round(total_ascend_hours, 1),
            },
        }

    def chat(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Shortcut for inference chat completion."""
        return self.inference.chat(prompt, model=model, stream=stream, **kwargs)

    def list_models(self) -> List[Dict[str, Any]]:
        """Shortcut for listing supported frontier models."""
        return self.tokenplan.list_models()
