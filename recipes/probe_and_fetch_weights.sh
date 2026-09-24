#!/bin/bash
# probe_and_fetch_weights.sh — run INSIDE the InkStone dev machine.
# 1) Measures download speed from candidate mirrors (10 s probe each).
# 2) Downloads the model weights to /data with the fastest source, resumable.
set -u
MODEL_DIR=/data/models/Qwen3.8-27B-Uncensored
HF_REPO=orcarouter/Qwen3.8-27B-Uncensored
PROBE_SIZE=$((10*1024*1024))   # 10 MB probe

speed_test() {
  local name="$1" url="$2"
  local t0 t1 bytes
  t0=$(date +%s.%N)
  bytes=$(curl -sL --max-time 10 -r 0-$((PROBE_SIZE-1)) -o /dev/null -w '%{size_download}' "$url" 2>/dev/null || echo 0)
  t1=$(date +%s.%N)
  local dt bps mbps
  dt=$(echo "$t1 - $t0" | bc)
  if echo "$dt > 0.01" | bc -l | grep -q 1; then
    bps=$(echo "$bytes / $dt" | bc -l)
    mbps=$(echo "scale=2; $bps / 1048576" | bc)
    echo "$mbps"
  else
    echo "0"
  fi
}

echo "=== bandwidth probe (10 MB per source) ==="
# Real probes use known-large static objects:
V_ALIYUN=$(speed_test aliyun    "https://mirrors.aliyun.com/deepin-cd/20/deepin-desktop-community-20.9-amd64.iso")
V_HFMIRROR=$(speed_test hfmirror "https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors")
V_MODELSCOPE=$(speed_test modelscope "https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/master/model-00001-of-00004.safetensors")
V_HF=$(speed_test huggingface "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors")

printf "aliyun:      %s MB/s\nhf-mirror:   %s MB/s\nmodelscope:  %s MB/s\nhuggingface: %s MB/s\n" \
  "$V_ALIYUN" "$V_HFMIRROR" "$V_MODELSCOPE" "$V_HF"

mkdir -p "$MODEL_DIR"

echo "=== starting weight download (resumable) ==="
# hf-mirror and huggingface both work with huggingface-cli; modelscope has its own CLI.
if echo "$V_MODELSCOPE > $V_HFMIRROR" | bc -l | grep -q 1; then
  echo "-- using modelscope CLI --"
  # ModelScope mirrors many HF repos 1:1; if this repo is absent it errors fast.
  export MODELSCOPE_CACHE=/data/models/_ms_cache
  modelscope download --model "$HF_REPO" --local_dir "$MODEL_DIR" || {
    echo "modelscope failed, falling back to hf-mirror"
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR" --resume-download
  }
else
  echo "-- using huggingface-cli via hf-mirror --"
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR" --resume-download
fi

echo "=== verifying ==="
du -sh "$MODEL_DIR"
ls -lh "$MODEL_DIR" | head -20
echo "Done. Serve with: /root/serve.sh"
