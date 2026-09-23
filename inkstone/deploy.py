"""OSS Model deployment generator and automation helper for dev machines."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

SUPPORTED_RECIPES = {
    "qwen3.8-uncensored": {
        "repo_id": "orcarouter/Qwen3.8-27B-Uncensored",
        "served_name": "qwen3.8-27b-uncensored",
        "default_dir": "/data/models/Qwen3.8-27B-Uncensored",
        "max_model_len": 32768,
        "gpu_util": 0.92,
        "dtype": "bfloat16",
        "description": "OrcaRouter Qwen 3.8 27B Uncensored (0% refusal, lowest KL loss)",
    },
    "cyberstrike-35b": {
        "repo_id": "huihui-ai/Huihui-CyberStrike-OffSec-35B-abliterated",
        "served_name": "cyberstrike-35b",
        "default_dir": "/data/models/Huihui-CyberStrike-OffSec-35B",
        "max_model_len": 65536,
        "gpu_util": 0.90,
        "dtype": "bfloat16",
        "description": "Huihui CyberStrike OffSec 35B (Fine-tuned on 500k+ offensive security samples)",
    },
}


class ModelDeployer:
    """Generates execution scripts and recipes for running self-hosted OSS models on A100 dev machines."""

    @staticmethod
    def generate_setup_script(recipe_name: str = "qwen3.8-uncensored", port: int = 8000, with_tunnel: bool = True) -> str:
        recipe = SUPPORTED_RECIPES.get(recipe_name)
        if not recipe:
            raise ValueError(f"Unknown recipe '{recipe_name}'. Supported: {list(SUPPORTED_RECIPES.keys())}")

        repo_id = recipe["repo_id"]
        served_name = recipe["served_name"]
        model_dir = recipe["default_dir"]
        max_len = recipe["max_model_len"]
        gpu_util = recipe["gpu_util"]
        dtype = recipe["dtype"]

        tunnel_block = ""
        if with_tunnel:
            tunnel_block = textwrap.dedent(f"""\
                # Setup Cloudflare Tunnel for public HTTPS endpoint
                echo "[+] Setting up tunnel for public HTTPS API endpoint..."
                mkdir -p /data/bin
                ARCH="$(uname -m)"
                if [ ! -f /data/bin/cloudflared ]; then
                    if [ "${{ARCH}}" = "x86_64" ]; then
                        curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /data/bin/cloudflared
                    elif [ "${{ARCH}}" = "aarch64" ]; then
                        curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o /data/bin/cloudflared
                    fi
                    chmod +x /data/bin/cloudflared
                fi
                /data/bin/cloudflared tunnel --url "http://127.0.0.1:{port}" --no-autoupdate > /tmp/tunnel.log 2>&1 &
                echo "[*] Waiting for public endpoint URL..."
                for i in {{1..30}}; do
                    TUNNEL_URL=$(grep -o 'https://[-a-z0-9]*\\.trycloudflare\\.com' /tmp/tunnel.log | head -n 1 || true)
                    if [ -n "${{TUNNEL_URL}}" ]; then
                        echo "=========================================================="
                        echo " [✓] PUBLIC API ENDPOINT OPENED:"
                        echo " Base URL: ${{TUNNEL_URL}}/v1"
                        echo " Completions: ${{TUNNEL_URL}}/v1/chat/completions"
                        echo "=========================================================="
                        break
                    fi
                    sleep 1
                done
            """)

        script = textwrap.dedent(f"""\
            #!/usr/bin/env bash
            # Auto-generated deployment script for {repo_id}
            # Target Hardware: Nvidia A100-1-80G on Intern InkStone
            set -euo pipefail

            echo "[+] Starting deployment of {repo_id}..."

            # 1. Ensure persistent directory structure on /data
            MODEL_DIR="{model_dir}"
            mkdir -p "${{MODEL_DIR}}"

            # 2. Configure mirror for fast downloads in China regions
            export HF_ENDPOINT="https://hf-mirror.com"
            export PYTHONUNBUFFERED=1

            # 3. Install/upgrade vLLM and huggingface_hub
            echo "[+] Checking/installing vLLM and huggingface_hub..."
            pip install -q -U huggingface_hub vllm

            # 4. Download weights to persistent /data volume if not already downloaded
            if [ -z "$(ls -A "${{MODEL_DIR}}" 2>/dev/null)" ]; then
                echo "[+] Downloading weights for {repo_id}..."
                huggingface-cli download "{repo_id}" \\
                    --local-dir "${{MODEL_DIR}}" \\
                    --local-dir-use-symlinks False
            else
                echo "[+] Weights already exist at ${{MODEL_DIR}}, skipping download."
            fi

            {tunnel_block}

            # 5. Launch vLLM OpenAI-Compatible API Server
            echo "[+] Launching vLLM on port {port}..."
            exec python3 -m vllm.entrypoints.openai.api_server \\
                --model "${{MODEL_DIR}}" \\
                --served-model-name "{served_name}" \\
                --tensor-parallel-size 1 \\
                --gpu-memory-utilization {gpu_util} \\
                --max-model-len {max_len} \\
                --dtype {dtype} \\
                --host 0.0.0.0 \\
                --port {port}
        """)
        return script

    @staticmethod
    def save_recipe(recipe_name: str, output_path: str | Path, port: int = 8000, with_tunnel: bool = True) -> Path:
        script = ModelDeployer.generate_setup_script(recipe_name, port=port, with_tunnel=with_tunnel)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(script, encoding="utf-8")
        p.chmod(0o755)
        return p
