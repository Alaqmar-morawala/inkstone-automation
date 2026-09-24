# Intern InkStone Automation Suite & SDK

A lightweight Python SDK and CLI tool for managing **Intern InkStone (*书生·端砚*) / Intern-AI / OpenXLab** cloud resources.

---

## Capabilities

* **TokenPlan Quotas & Monitoring**: Inspect token points balance (10 墨点 / ~200M tokens), 5-hour and 7-day rolling window recovery timers.
* **Frontier Model Inference**: Query 10 hosted frontier models (`qwen3.8-27b`, `deepseek-v4-pro-0813`, `glm-5.3`, `kimi-k2.6`, etc.) with streaming support over OpenAI-compatible endpoints.
* **API Key Management**: Create, list, disable, and revoke TokenPlan API keys.
* **Dev Machine & GPU Lifecycle Automation**: Spin up, list, start, stop, and delete **Nvidia A100-1-80G**, **Ascend 910B-1-64G**, and CPU environments with persistent `/data` storage.
* **OSS Model Deployer**: One-command recipe generation to download and serve self-hosted open-source models (such as `orcarouter/Qwen3.8-27B-Uncensored`) using vLLM on A100 instances.

---

## Directory Structure

```
/home/alaqmar/inkstone/
├── config.json                        # Pre-configured session credentials & endpoints
├── bin/
│   └── inkstone                       # Direct CLI executable wrapper
├── inkstone/                          # Python package
│   ├── __init__.py                    # Public SDK exports
│   ├── config.py                      # Config loading & environment parser
│   ├── auth.py                        # JWT session validation & header manager
│   ├── tokenplan.py                   # Credit balance, models & API key manager
│   ├── inference.py                   # Chat completions & streaming client
│   ├── compute.py                     # Dev machine lifecycle manager
│   ├── deploy.py                      # OSS model deployment recipes (vLLM)
│   ├── client.py                      # Consolidated client facade
│   └── cli.py                         # Command-line interface
├── recipes/
│   └── deploy_qwen38_uncensored.sh   # Standalone runner for Qwen 3.8 27B Uncensored
├── tests/
│   └── test_inkstone.py               # Automated verification suite
├── requirements.txt                   # Dependencies (httpx>=0.28.0)
└── setup.py                           # Package installer
```

---

## Quickstart (CLI)

Add `bin/inkstone` to your PATH or run it directly:

```bash
# 1. View pooled status across all configured accounts
/home/alaqmar/inkstone/bin/inkstone status --all

# 2. Check active account balance and quota (Tokens & Compute)
/home/alaqmar/inkstone/bin/inkstone status

# 3. List and switch between registered accounts
/home/alaqmar/inkstone/bin/inkstone accounts list
/home/alaqmar/inkstone/bin/inkstone accounts switch acc2

# 4. Run a command on a specific account without switching default
/home/alaqmar/inkstone/bin/inkstone --account acc2 status

# 5. View all 10 available cloud models
/home/alaqmar/inkstone/bin/inkstone models

# 6. Chat with a cloud model (with streaming)
/home/alaqmar/inkstone/bin/inkstone chat -m qwen3.8-27b --stream "Explain race conditions in Web3 smart contracts."

# 7. List and manage TokenPlan API keys
/home/alaqmar/inkstone/bin/inkstone keys list
/home/alaqmar/inkstone/bin/inkstone keys create --name my-new-key

# 8. Check available GPU compute flavors & rates
/home/alaqmar/inkstone/bin/inkstone compute flavors

# 9. List and control Dev Machines
/home/alaqmar/inkstone/bin/inkstone compute list
/home/alaqmar/inkstone/bin/inkstone compute create --gpu a100 --hours 2 --name test-a100
/home/alaqmar/inkstone/bin/inkstone compute url <machine_id>
/home/alaqmar/inkstone/bin/inkstone compute stop <machine_id>

# 10. Generate deployment script for Qwen 3.8 27B Uncensored
/home/alaqmar/inkstone/bin/inkstone deploy qwen3.8-uncensored
```

---

## Python SDK Usage

### 1. Account Status & Quotas

```python
from inkstone import InkStoneClient

client = InkStoneClient()
status = client.get_full_status()

print("Available Tokens:", status["tokens"]["approx_tokens_formatted"])
print("Compute Points:", status["compute"]["remaining_points"])
print("A100 Hours Left:", status["compute"]["a100_hours_remaining"])
```

### 2. Model Inference

```python
from inkstone import InkStoneClient

client = InkStoneClient()

# Non-streaming call
response = client.chat(
    "List three common authorization vulnerabilities in REST APIs.",
    model="deepseek-v4-pro-0813",
    temperature=0.3
)
print("Thinking:", response.get("reasoning_content"))
print("Answer:", response["content"])

# Streaming call
for chunk in client.chat("Summarize AES-GCM.", model="qwen3.8-27b", stream=True):
    print(chunk, end="", flush=True)
```

### 3. GPU Dev Machine Automation

```python
from inkstone import InkStoneClient

client = InkStoneClient()

# Check compute hardware options
flavors = client.compute.list_flavors()

# Create an Nvidia A100 instance for 3 hours
machine = client.compute.create_machine(name="research-gpu", flavor="a100", hours=3)
machine_id = machine["id"]

# Retrieve web IDE URLs (VSCode code-server & JupyterLab)
urls = client.compute.get_ide_urls(machine_id)
print("VSCode URL:", urls["codeserver"])
print("JupyterLab URL:", urls["jupyterlab"])

# Crucial: Stop the instance when done to freeze point deductions
client.compute.stop_machine(machine_id)
```

---

## Deploying `orcarouter/Qwen3.8-27B-Uncensored` on A100

1. Create an A100 machine:
   ```bash
   /home/alaqmar/inkstone/bin/inkstone compute create --gpu a100 --hours 4 --name qwen-server
   ```
2. Retrieve the IDE URL:
   ```bash
   /home/alaqmar/inkstone/bin/inkstone compute url <machine_id>
   ```
3. Open the VSCode terminal in your browser and execute the runner recipe:
   ```bash
   bash /home/alaqmar/inkstone/recipes/deploy_qwen38_uncensored.sh
   ```
   *(Or copy the script generated by `inkstone deploy qwen3.8-uncensored` into the machine).*
4. Weights will download to persistent storage at `/data/models/Qwen3.8-27B-Uncensored` and vLLM will serve an OpenAI-compatible endpoint on port 8000.

---

## Multi-Account Token Proxy (for ZCode & other harnesses)

`inkstone proxy` runs a local gateway that pools **all accounts' 200M-token quotas** and routes each request to the account with the most remaining 5h-window credit, dodging the per-account rate limits (50 req/min, 2M tok/min) and the 5h/7d sliding windows.

```bash
inkstone proxy --port 8799                 # all accounts, no auth (localhost only)
inkstone proxy --proxy-key mysecret        # harnesses must send Bearer mysecret
inkstone proxy --accounts acc1,acc2        # pool a subset
```

Endpoints:

| Endpoint | Protocol |
|---|---|
| `POST /v1/chat/completions` | OpenAI (streaming supported) |
| `POST /v1/messages` | Anthropic (streaming supported) |
| `GET /v1/models` | Model catalog |
| `GET /quota` | Live per-account balances, RPM/TPM, routing counters |

Every response carries an `x-inkstone-account` header telling you which account served it.

**Connect ZCode (OpenAI protocol):**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8799/v1
export OPENAI_API_KEY=anything          # or your --proxy-key
```

**Connect ZCode (Anthropic protocol):**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8799
export ANTHROPIC_AUTH_TOKEN=anything    # or your --proxy-key
```

Routing policy: skip accounts in cooldown (auto-triggered by upstream 429/5xx, 60s), stay under local RPM/TPM windows, then pick the account with the most remaining 5h credit. Accounts whose management token expired (balance unknown) still participate via their API key. Failover to the next-best account is transparent.

---

## Point Cost & Rate Reference

* **TokenPlan Base Unit**: 1 墨点 (Ink Point) ≈ 20,000,000 tokens
* **5-Hour Limit**: 10 墨点 (recovers continuously every 5 hours)
* **7-Day Limit**: 50 墨点
* **Nvidia A100 80GB**: 60 算力点 / hour
* **Ascend 910B 64GB**: 53 算力点 / hour
* **CPU 4C-16G**: 23 算力点 / hour
* **Storage**: Persistent `/data` volume remains intact after machine stoppage.
