#!/usr/bin/env bash
# Standalone deployment & API endpoint launcher for orcarouter/Qwen3.8-27B-Uncensored
# Designed for Nvidia A100-1-80G dev machines on Intern InkStone
set -euo pipefail

MODEL_ID="orcarouter/Qwen3.8-27B-Uncensored"
SERVED_NAME="qwen3.8-27b-uncensored"
TARGET_DIR="/data/models/Qwen3.8-27B-Uncensored"
PORT="8000"

echo "=========================================================="
echo " Starting Deployment of ${MODEL_ID}"
echo " Target Storage: ${TARGET_DIR}"
echo "=========================================================="

mkdir -p "${TARGET_DIR}" /data/bin
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1

# 1. Install dependencies
echo "[1/4] Ensuring required dependencies (vllm, huggingface_hub)..."
pip install -q -U huggingface_hub vllm

# 2. Setup Cloudflare Tunnel binary for opening public HTTPS API endpoint
echo "[2/4] Setting up tunnel binary for public API access..."
ARCH="$(uname -m)"
if [ ! -f /data/bin/cloudflared ]; then
    if [ "${ARCH}" = "x86_64" ]; then
        curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /data/bin/cloudflared
    elif [ "${ARCH}" = "aarch64" ]; then
        curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o /data/bin/cloudflared
    fi
    chmod +x /data/bin/cloudflared
fi

# 3. Download weights to persistent /data volume if missing
if [ -z "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "[3/4] Downloading model weights for ${MODEL_ID}..."
    huggingface-cli download "${MODEL_ID}" \
        --local-dir "${TARGET_DIR}" \
        --local-dir-use-symlinks False
else
    echo "[3/4] Model weights verified in persistent storage: ${TARGET_DIR}"
fi

# 4. Start Cloudflare Tunnel in background to open public API endpoint
echo "[4/4] Starting public HTTPS tunnel and vLLM server..."
/data/bin/cloudflared tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate > /tmp/tunnel.log 2>&1 &
TUNNEL_PID=$!

echo "[*] Waiting for public HTTPS endpoint URL..."
for i in {1..30}; do
    TUNNEL_URL=$(grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' /tmp/tunnel.log | head -n 1 || true)
    if [ -n "${TUNNEL_URL}" ]; then
        echo "=========================================================="
        echo " [✓] PUBLIC API ENDPOINT OPENED:"
        echo " Base URL: ${TUNNEL_URL}/v1"
        echo " Completions: ${TUNNEL_URL}/v1/chat/completions"
        echo " Example curl:"
        echo "   curl ${TUNNEL_URL}/v1/chat/completions \\"
        echo "     -H 'Content-Type: application/json' \\"
        echo "     -d '{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
        echo "=========================================================="
        break
    fi
    sleep 1
done

# 5. Launch vLLM OpenAI-Compatible API Server in foreground
echo "[*] Starting vLLM engine on port ${PORT}..."
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${TARGET_DIR}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "${PORT}"
