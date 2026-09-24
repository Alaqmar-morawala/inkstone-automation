"""Compute and Dev Machine lifecycle manager for Nvidia A100, Ascend 910B, and CPU environments."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from .auth import AuthManager
from .config import InkStoneConfig

# Default mapping for hardware flavors
FLAVOR_CONFIG = {
    "a100": {
        "resource_id": 15,
        "config_name": "NvidiaA100-1-80G",
        "default_mirror_id": 1,  # cuda 12.6 / pytorch 2.4.0 / python 3.11
        "hourly_cost": 60,
        "vram": "80GiB",
        "description": "Nvidia A100 SXM 80GB GPU (4 vCPU, 16GB RAM)",
    },
    "910b": {
        "resource_id": 12,
        "config_name": "Ascend910B-1-64G",
        "default_mirror_id": 11,  # cann 8.3rc2 / pytorch 2.8.0 / python 3.11 arm64
        "hourly_cost": 53,
        "vram": "64GiB",
        "description": "Huawei Ascend 910B 64GB NPU (4 vCPU, 16GB RAM, ARM64)",
    },
    "cpu": {
        "resource_id": 11,
        "config_name": "CPU-4C-16G",
        "default_mirror_id": 9,  # pytorch 2.4.0 / python 3.11
        "hourly_cost": 23,
        "vram": "0GiB",
        "description": "Standard CPU (4 vCPU, 16GB RAM)",
    },
}


class ComputeManager:
    """Manages cloud dev machines, GPU instances, IDE endpoints, and point costs."""

    def __init__(self, config: Optional[InkStoneConfig] = None, auth: Optional[AuthManager] = None):
        self.config = config or InkStoneConfig.load()
        self.auth = auth or AuthManager(self.config)
        self.base_url = self.config.discovery_url

    def get_user_resource(self) -> Dict[str, Any]:
        """Query user computility points balance and notebook limits."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/user_resource"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to query compute resources: {data.get('msg')}")

        payload = data.get("data", {})
        total_pts = payload.get("computility_point_total", 0)
        used_pts = payload.get("computility_point_used", 0)
        remaining_pts = total_pts - used_pts

        return {
            "total_points": total_pts,
            "used_points": used_pts,
            "remaining_points": remaining_pts,
            "a100_hours_remaining": round(remaining_pts / 60.0, 1),
            "ascend_hours_remaining": round(remaining_pts / 53.0, 1),
            "notebooks_total": payload.get("notebook_total", 10),
            "notebooks_used": payload.get("notebook_used", 0),
            "raw": payload,
        }

    def list_flavors(self) -> List[Dict[str, Any]]:
        """List available compute hardware specifications and point rates."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/computility_resources?category=notebook"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("data", {}).get("list", [])

    def list_mirrors(self) -> List[Dict[str, Any]]:
        """List official base container images."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/mirrors?category=notebook"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("data", {}).get("list", [])

    def list_machines(self, page: int = 1, page_size: int = 20) -> List[Dict[str, Any]]:
        """List existing dev machines."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/list"
        headers = self.auth.get_portal_headers()
        params = {"page": page, "pageSize": page_size}

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        return data.get("data", {}).get("list", [])

    def create_machine(
        self,
        name: str,
        flavor: str = "a100",
        hours: int = 2,
        mirror_id: Optional[int] = None,
        custom_image: bool = False,
    ) -> Dict[str, Any]:
        """Create a new development machine.

        Args:
            custom_image: If True, treat mirror_id as a custom image ID and
                          set mirror_source to 'custom' so the platform uses
                          a user-built container image instead of an official one.
        """
        flavor_key = flavor.lower().replace("-", "")
        if flavor_key not in FLAVOR_CONFIG:
            raise ValueError(f"Unknown flavor '{flavor}'. Choose from: {list(FLAVOR_CONFIG.keys())}")

        fcfg = FLAVOR_CONFIG[flavor_key]
        chosen_mirror_id = mirror_id or fcfg["default_mirror_id"]
        duration_seconds = hours * 3600

        payload = {
            "name": name,
            "mirror_id": chosen_mirror_id,
            "mirror_source": "custom" if custom_image else "official",
            "resource_id": fcfg["resource_id"],
            "run_duration": duration_seconds,
            "dataset_selected": "",
        }

        url = f"{self.base_url}/api/ml/v1/dev-machine/create"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to create dev machine: {data.get('msg')} (code: {data.get('code')})")

        return data.get("data", {})

    def _resolve_pod_name(self, machine_id: int | str, pod_name: Optional[str] = None) -> str:
        if pod_name:
            return pod_name
        for m in self.list_machines():
            if str(m.get("id")) == str(machine_id):
                return m.get("pod_name") or ""
        return ""

    def start_machine(self, machine_id: int | str, hours: int = 2, pod_name: Optional[str] = None) -> bool:
        """Start or resume a stopped dev machine."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/start"
        headers = self.auth.get_portal_headers()
        pname = self._resolve_pod_name(machine_id, pod_name)
        payload = {
            "id": int(machine_id),
            "pod_name": pname,
            "run_duration": hours * 3600,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data.get("code") == 0

    def stop_machine(self, machine_id: int | str, pod_name: Optional[str] = None) -> bool:
        """Stop a running dev machine to freeze point deduction."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/stop"
        headers = self.auth.get_portal_headers()
        pname = self._resolve_pod_name(machine_id, pod_name)
        payload = {"id": int(machine_id), "pod_name": pname}

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data.get("code") == 0

    def delete_machine(self, machine_id: int | str, pod_name: Optional[str] = None) -> bool:
        """Permanently delete a dev machine and release allocated resources."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/delete"
        headers = self.auth.get_portal_headers()
        pname = self._resolve_pod_name(machine_id, pod_name)
        payload = {"id": int(machine_id), "pod_name": pname}

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data.get("code") == 0

    def get_ide_urls(self, machine_id: int | str, pod_name: Optional[str] = None) -> Dict[str, str]:
        """Fetch the web-based IDE URLs (VSCode code-server & JupyterLab) for a running machine."""
        resolved_pod_name = self._resolve_pod_name(machine_id, pod_name)

        url = f"{self.base_url}/api/ml/v1/dev-machine/enter"
        headers = self.auth.get_portal_headers()
        payload = {
            "id": int(machine_id),
            "pod_name": resolved_pod_name or "",
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Machine not accessible: {data.get('msg')} (code: {data.get('code')})")

        return data.get("data", {})
