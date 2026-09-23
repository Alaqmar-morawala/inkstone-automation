"""Configuration manager for Intern InkStone automation with multi-account support."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class InkStoneConfig:
    active_account: str = "acc1"
    accounts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    uaa_token: str = ""
    ssouid: str = ""
    api_key: str = ""
    acw_tc: str = ""
    discovery_url: str = "https://discovery.intern-ai.org.cn"
    api_base_url: str = "https://discovery-api.intern-ai.org.cn/v1"
    anthropic_base_url: str = "https://discovery-api.intern-ai.org.cn"
    default_model: str = "qwen3.8-27b"

    @classmethod
    def load(cls, path: Optional[str | Path] = None, account: Optional[str] = None) -> InkStoneConfig:
        config_file = Path(path) if path else DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}

        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read {config_file}: {e}")

        accounts = data.get("accounts", {})
        active_account = account or os.environ.get("INKSTONE_ACCOUNT") or data.get("active_account", "acc1")

        # Resolve credentials for the chosen active account
        acc_data = accounts.get(active_account, {})

        uaa_token = os.environ.get("INKSTONE_UAA_TOKEN", acc_data.get("uaa_token") or data.get("uaa_token", ""))
        ssouid = os.environ.get("INKSTONE_SSOUID", acc_data.get("ssouid") or data.get("ssouid", ""))
        api_key = os.environ.get("INKSTONE_API_KEY", acc_data.get("api_key") or data.get("api_key", ""))
        acw_tc = os.environ.get("INKSTONE_ACW_TC", acc_data.get("acw_tc") or data.get("acw_tc", ""))
        discovery_url = os.environ.get("INKSTONE_DISCOVERY_URL", data.get("discovery_url", "https://discovery.intern-ai.org.cn"))
        api_base_url = os.environ.get("INKSTONE_API_BASE_URL", data.get("api_base_url", "https://discovery-api.intern-ai.org.cn/v1"))
        anthropic_base_url = os.environ.get("INKSTONE_ANTHROPIC_BASE_URL", data.get("anthropic_base_url", "https://discovery-api.intern-ai.org.cn"))
        default_model = os.environ.get("INKSTONE_DEFAULT_MODEL", data.get("default_model", "qwen3.8-27b"))

        return cls(
            active_account=active_account,
            accounts=accounts,
            uaa_token=uaa_token,
            ssouid=ssouid,
            api_key=api_key,
            acw_tc=acw_tc,
            discovery_url=discovery_url.rstrip("/"),
            api_base_url=api_base_url.rstrip("/"),
            anthropic_base_url=anthropic_base_url.rstrip("/"),
            default_model=default_model,
        )

    def switch_account(self, account_name: str) -> None:
        """Switch active account and update active credentials."""
        if account_name not in self.accounts:
            raise ValueError(f"Account '{account_name}' not found. Available: {list(self.accounts.keys())}")
        self.active_account = account_name
        acc_data = self.accounts[account_name]
        self.uaa_token = acc_data.get("uaa_token", "")
        self.ssouid = acc_data.get("ssouid", "")
        self.api_key = acc_data.get("api_key", "")
        self.acw_tc = acc_data.get("acw_tc", "")

    def save(self, path: Optional[str | Path] = None) -> None:
        config_file = Path(path) if path else DEFAULT_CONFIG_PATH
        config_file.parent.mkdir(parents=True, exist_ok=True)
        # Update accounts dict with current active credentials
        if self.active_account and self.active_account in self.accounts:
            self.accounts[self.active_account]["uaa_token"] = self.uaa_token
            self.accounts[self.active_account]["ssouid"] = self.ssouid
            self.accounts[self.active_account]["api_key"] = self.api_key
            self.accounts[self.active_account]["acw_tc"] = self.acw_tc

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
