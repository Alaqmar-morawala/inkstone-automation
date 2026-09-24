"""Custom container image management for InkStone dev machines.

Supports building custom Docker images on InkStone's Harbor registry with
pre-installed dependencies via pip/apt (quick mode) or full Dockerfile.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .auth import AuthManager
from .config import InkStoneConfig


class ImageManager:
    """Manages custom container images (mirrors) for dev machines."""

    def __init__(self, config: Optional[InkStoneConfig] = None, auth: Optional[AuthManager] = None):
        self.config = config or InkStoneConfig.load()
        self.auth = auth or AuthManager(self.config)
        self.base_url = self.config.discovery_url

    def list_images(
        self,
        page: int = 1,
        size: int = 20,
        keyword: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        sort: str = "update_time_desc",
    ) -> Dict[str, Any]:
        """List custom images with optional filtering.

        Args:
            status: Filter by building_status (pending, building, success, failed).
            category: Filter by use scene (notebook, training).
            sort: Sort order (update_time_desc, update_time_asc).

        Returns:
            Dict with 'list', 'total', and 'max_count' keys.
        """
        url = f"{self.base_url}/api/ml/v1/mirrors"
        headers = self.auth.get_portal_headers()
        params: Dict[str, Any] = {"page": page, "size": size, "sort_type": sort}
        if keyword:
            params["keyword"] = keyword
        if status:
            params["building_status"] = status
        if category:
            params["category"] = category

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to list images: {data.get('msg')}")

        payload = data.get("data", {})
        return {
            "list": payload.get("list", []),
            "total": len(payload.get("list", [])),
            "max_count": payload.get("max_count", 0),
        }

    def get_image(self, image_id: int | str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a custom image."""
        url = f"{self.base_url}/api/ml/v1/mirrors/{image_id}"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            return None
        return data.get("data")

    def get_limit(self) -> int:
        """Get the maximum number of custom images allowed per account."""
        url = f"{self.base_url}/api/ml/v1/mirrors/limit"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") == 0 and data.get("data"):
            return data["data"].get("limit", 5)
        return 5

    def create_image_quick(
        self,
        name: str,
        base_image_id: int = 1,
        resource_id: int = 15,
        category: str = "notebook",
        packages: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Create a custom image using quick mode (pip/apt package lists).

        Args:
            name: Display name for the image.
            base_image_id: Base mirror ID (1=A100 GPU, 9=CPU, 11=910B NPU).
            resource_id: Resource config ID (15=A100, 12=910B, 11=CPU).
            category: Use scene ('notebook' or 'training').
            packages: Dict mapping install method to package list,
                      e.g. {'pip': ['vllm>=0.8.0', 'transformers'], 'apt': ['htop']}.

        Returns:
            API response data.
        """
        install_set = []
        if packages:
            for method, pkg_list in packages.items():
                parsed = []
                for pkg in pkg_list:
                    if "==" in pkg:
                        idx = pkg.rindex("==")
                        parsed.append({"name": pkg[:idx], "version": pkg[idx + 2:]})
                    else:
                        parsed.append({"name": pkg})
                install_set.append({"method": method, "set": parsed})

        payload = {
            "category": category,
            "name": name,
            "basic_mirror_id": base_image_id,
            "resource_id": resource_id,
            "build_config_method": "quick",
            "quick_install_set": install_set,
        }

        return self._post_create(payload)

    def create_image_dockerfile(
        self,
        name: str,
        dockerfile_content: str,
        base_image_id: int = 1,
        resource_id: int = 15,
        category: str = "notebook",
        include_from: bool = True,
    ) -> Dict[str, Any]:
        """Create a custom image using a Dockerfile.

        Args:
            name: Display name for the image.
            dockerfile_content: Dockerfile body (RUN, ENV, COPY instructions).
            base_image_id: Base mirror ID (1=A100 GPU, 9=CPU, 11=910B NPU).
            resource_id: Resource config ID (15=A100, 12=910B, 11=CPU).
            category: Use scene ('notebook' or 'training').
            include_from: If True, prepend FROM <base_image_url> automatically.

        Returns:
            API response data.
        """
        if include_from:
            base_url = self._get_base_image_url(base_image_id)
            if base_url:
                full_dockerfile = f"FROM {base_url}\n{dockerfile_content}"
            else:
                full_dockerfile = dockerfile_content
        else:
            full_dockerfile = dockerfile_content

        payload = {
            "category": category,
            "name": name,
            "basic_mirror_id": base_image_id,
            "resource_id": resource_id,
            "build_config_method": "dockerfile",
            "dockerfile_content": full_dockerfile,
        }

        return self._post_create(payload)

    def update_image(self, image_id: int | str, **kwargs: Any) -> Dict[str, Any]:
        """Update an existing custom image (re-triggers build)."""
        url = f"{self.base_url}/api/ml/v1/mirrors/{image_id}"
        headers = self.auth.get_portal_headers()
        headers["Content-Type"] = "application/json"

        with httpx.Client(timeout=15.0) as client:
            resp = client.put(url, headers=headers, json=kwargs)
            resp.raise_for_status()
            return resp.json()

    def delete_image(self, image_id: int | str) -> bool:
        """Delete a custom image."""
        url = f"{self.base_url}/api/ml/v1/mirrors/{image_id}"
        headers = self.auth.get_portal_headers()

        with httpx.Client(timeout=15.0) as client:
            resp = client.delete(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("code") == 0

    def _post_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal: POST to create a new custom image."""
        url = f"{self.base_url}/api/ml/v1/mirrors"
        headers = self.auth.get_portal_headers()
        headers["Content-Type"] = "application/json"

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Image creation failed: {data.get('msg')}")

        return data.get("data", {})

    def _get_base_image_url(self, mirror_id: int) -> Optional[str]:
        """Resolve a base mirror ID to its container registry URL."""
        base_mirrors = {
            1: "discovery-harbor-p.intern-ai.org.cn:32080/cloud/notebook-gpu-a100:latest",
            9: "discovery-harbor-d.intern-ai.org.cn:8080/discovery/notebook-cpu:20260519",
            11: "discovery-harbor-d.intern-ai.org.cn:8080/discovery/notebook-npu-a2cann-910b:20260519",
        }
        return base_mirrors.get(mirror_id)

    def list_base_images(self) -> List[Dict[str, Any]]:
        """List official base images that can be used as FROM targets."""
        url = f"{self.base_url}/api/ml/v1/dev-machine/mirrors"
        headers = self.auth.get_portal_headers()
        params = {"category": "notebook"}

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        return data.get("data", {}).get("list", [])

    def wait_for_build(
        self,
        image_id: int | str,
        poll_interval: int = 15,
        max_wait: int = 1800,
        callback: Any = None,
    ) -> str:
        """Poll until a custom image build completes or fails.

        Returns:
            Final building_status ('success' or 'failed').
        """
        import time

        elapsed = 0
        while elapsed < max_wait:
            info = self.get_image(image_id)
            if info is None:
                raise RuntimeError(f"Image {image_id} not found")

            status = info.get("building_status", "unknown")
            if callback:
                callback(status, elapsed, info)

            if status in ("success", "failed"):
                return status

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Image build did not complete within {max_wait}s")
