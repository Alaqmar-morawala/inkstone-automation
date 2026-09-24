"""Multi-account token proxy for InkStone frontier models.

Sits between a harness (ZCode, IDE plugins, CLI tools) and the InkStone
inference API. Speaks both OpenAI (/v1/chat/completions) and Anthropic
(/v1/messages) protocols, and routes every request to the account with the
most remaining 5h-window credit, so the pooled 200M-token quotas are used
evenly and 429 rate limits are avoided.

Routing policy (per request):
  1. Skip accounts in cooldown (recent 429 / upstream failure).
  2. Soft-skip accounts whose local RPM/TPM window is near the platform
     limit (50 req/min, 2M tokens/min).
  3. Rank by remaining 5h-window credit (fetched from the balance API every
     ~45s). Accounts with an unknown balance (expired management token) rank
     mid-pack so they still get used; known-drained accounts rank last.
  4. On 429 or upstream 5xx: cooldown the account for 60s and fail over to
     the next-best account.
"""

# NOTE: no `from __future__ import annotations` here - FastAPI resolves
# parameter annotations at decoration time and would fail to resolve
# closure-local names (e.g. Request) if they were lazy strings.
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx

log = logging.getLogger("inkstone.proxy")

DISCOVERY_BASE = "https://discovery.intern-ai.org.cn"
INFERENCE_BASE = "https://discovery-api.intern-ai.org.cn"

FALLBACK_MODELS = [
    "Agents-A1", "Atria-Dawn-Preview", "deepseek-v4-flash-0731",
    "deepseek-v4-flash-vision", "deepseek-v4-pro-0813", "glm-5.3",
    "intern-s2", "kimi-k2.6", "minimax-m3", "qwen3.8-27b",
]

# Assumed 5h-remaining credit for accounts whose balance is unknown.
UNKNOWN_BALANCE = 5.0


@dataclass
class AccountState:
    name: str
    api_key: str
    uaa_token: str = ""
    ssouid: str = ""

    remaining_5h: Optional[float] = None      # None = unknown
    remaining_7d: Optional[float] = None
    available_credits: Optional[float] = None
    token_expired: bool = False

    rpm: deque = field(default_factory=deque)          # request timestamps
    tpm: deque = field(default_factory=deque)          # (ts, tokens) pairs
    cooldown_until: float = 0.0
    last_balance_fetch: float = 0.0
    requests_served: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: int = 0

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def cooldown_left(self) -> float:
        return max(0.0, self.cooldown_until - time.time())

    def recent_rpm(self) -> int:
        cutoff = time.time() - 60.0
        while self.rpm and self.rpm[0] < cutoff:
            self.rpm.popleft()
        return len(self.rpm)

    def recent_tpm(self) -> int:
        cutoff = time.time() - 60.0
        while self.tpm and self.tpm[0][0] < cutoff:
            self.tpm.popleft()
        return sum(t for _, t in self.tpm)

    def record_usage(self, usage: dict) -> None:
        p = usage.get("prompt_tokens") or 0
        c = usage.get("completion_tokens") or 0
        self.tokens_in += p
        self.tokens_out += c
        self.tpm.append((time.time(), p + c))

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "remaining_5h": self.remaining_5h,
            "remaining_7d": self.remaining_7d,
            "available_credits": self.available_credits,
            "balance_known": self.remaining_5h is not None,
            "token_expired": self.token_expired,
            "cooldown_s": round(self.cooldown_left, 1),
            "rpm_recent": self.recent_rpm(),
            "tpm_recent": self.recent_tpm(),
            "requests_served": self.requests_served,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "errors": self.errors,
        }


class AccountPool:
    """Owns the accounts, balance refresh loop, and routing decision."""

    def __init__(self, accounts: list[AccountState], rpm_limit: int = 50,
                 tpm_limit: int = 2_000_000, cooldown_s: float = 60.0):
        self.accounts = accounts
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.cooldown_s = cooldown_s
        self.models: list[str] = list(FALLBACK_MODELS)
        self._models_fetched = False
        self._refresh_task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0))

    # ---------- balance ----------

    def _mgmt_headers(self, acc: AccountState) -> dict:
        return {
            "Cookie": f"ssouid={acc.ssouid}; uaa_token={acc.uaa_token}; "
                      f"uaa-token={acc.uaa_token}; token={acc.uaa_token}",
            "Authorization": f"Bearer {acc.uaa_token}",
            "token": acc.uaa_token,
            "uaa-token": acc.uaa_token,
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer": f"{DISCOVERY_BASE}/",
        }

    async def refresh_balance(self, acc: AccountState) -> None:
        if not acc.uaa_token:
            return
        try:
            r = await self._client.get(
                f"{DISCOVERY_BASE}/api/tokenplan/v1/credits/balance",
                headers=self._mgmt_headers(acc))
            d = r.json()
            data = d.get("data") if isinstance(d, dict) else None
            if data:
                acc.remaining_5h = float(data["usage_windows"]["5h"]["remaining_credits"])
                acc.remaining_7d = float(data["usage_windows"]["7d"]["remaining_credits"])
                acc.available_credits = float(data.get("available_credits", 0))
                acc.token_expired = False
            elif r.status_code == 401 or (isinstance(d, dict) and d.get("msgCode") == "A0211"):
                if not acc.token_expired:
                    log.warning("[%s] management token expired - routing without balance data",
                                acc.name)
                acc.token_expired = True
        except Exception as e:
            log.debug("[%s] balance refresh failed: %s", acc.name, e)
        finally:
            acc.last_balance_fetch = time.time()

    async def refresh_models(self) -> None:
        for acc in self.accounts:
            try:
                r = await self._client.get(
                    f"{INFERENCE_BASE}/v1/models",
                    headers={"Authorization": f"Bearer {acc.api_key}"})
                if r.status_code == 200:
                    ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
                    if ids:
                        self.models = sorted(ids)
                        self._models_fetched = True
                        return
            except Exception:
                continue

    async def _refresh_loop(self) -> None:
        while True:
            for acc in self.accounts:
                if time.time() - acc.last_balance_fetch > 45.0:
                    await self.refresh_balance(acc)
            if not self._models_fetched:
                await self.refresh_models()
            await asyncio.sleep(15.0)

    def start(self) -> None:
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    # ---------- routing ----------

    def _rank(self, acc: AccountState) -> tuple:
        remaining = acc.remaining_5h if acc.remaining_5h is not None else UNKNOWN_BALANCE
        # Lower tuple = better. Local RPM headroom dominates so we never
        # trip the platform's 50 req/min limit on one account.
        rpm_headroom = self.rpm_limit - 5 - acc.recent_rpm()
        return (0 if rpm_headroom > 0 else 1, -remaining, acc.recent_rpm(),
                -acc.requests_served)

    def pick(self, exclude: set[str] | None = None) -> Optional[AccountState]:
        exclude = exclude or set()
        candidates = [a for a in self.accounts
                      if a.name not in exclude and not a.in_cooldown]
        if not candidates:
            # Everyone exhausted/cooldown: ignore cooldown as last resort.
            candidates = [a for a in self.accounts if a.name not in exclude]
        if not candidates:
            return None
        return min(candidates, key=self._rank)

    def note_request(self, acc: AccountState) -> None:
        acc.rpm.append(time.time())
        acc.requests_served += 1

    def note_rate_limited(self, acc: AccountState) -> None:
        acc.cooldown_until = time.time() + self.cooldown_s
        acc.errors += 1
        log.warning("[%s] rate limited - cooldown %.0fs", acc.name, self.cooldown_s)
        asyncio.create_task(self.refresh_balance(acc))

    def note_error(self, acc: AccountState, status: int) -> None:
        acc.errors += 1
        if status >= 500:
            acc.cooldown_until = time.time() + 20.0

    def snapshot(self) -> dict:
        return {
            "accounts": [a.snapshot() for a in self.accounts],
            "models": self.models,
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
        }


class UpstreamError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"upstream {status}: {body[:200]}")


OPENAI_CHAT = f"{INFERENCE_BASE}/v1/chat/completions"
ANTHROPIC_MESSAGES = f"{INFERENCE_BASE}/v1/messages"


def _strip_proxy_headers(h: httpx.Headers) -> dict:
    return {k: v for k, v in h.items()
            if k.lower() not in ("content-length", "transfer-encoding",
                                 "connection", "content-encoding")}


async def attempt(pool: AccountPool, acc: AccountState, path: str,
                  body: dict, headers_extra: dict | None = None,
                  auth_style: str = "bearer") -> httpx.Response:
    """Send one non-streaming upstream request. Raises UpstreamError on
    retryable failures so the caller can fail over."""
    headers = {"Authorization": f"Bearer {acc.api_key}",
               "Content-Type": "application/json", "Accept": "application/json"}
    if auth_style == "anthropic":
        headers = {"x-api-key": acc.api_key,
                   "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json", "Accept": "application/json"}
    if headers_extra:
        headers.update(headers_extra)
    try:
        resp = await pool._client.post(path, json=body, headers=headers)
    except httpx.HTTPError as e:
        pool.note_error(acc, 599)
        raise UpstreamError(599, f"connection error: {e}") from e
    if resp.status_code == 429:
        pool.note_rate_limited(acc)
        raise UpstreamError(429, resp.text)
    if resp.status_code >= 500 or resp.status_code == 408:
        pool.note_error(acc, resp.status_code)
        raise UpstreamError(resp.status_code, resp.text)
    return resp


async def stream_attempt(pool: AccountPool, acc: AccountState, path: str,
                         body: dict, headers_extra: dict | None = None,
                         auth_style: str = "bearer") -> AsyncIterator[bytes]:
    """Streaming upstream generator. Validates the response status first so
    the caller can fail over before any bytes reach the client."""
    headers = {"Authorization": f"Bearer {acc.api_key}",
               "Content-Type": "application/json", "Accept": "text/event-stream"}
    if auth_style == "anthropic":
        headers = {"x-api-key": acc.api_key,
                   "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json", "Accept": "text/event-stream"}
    if headers_extra:
        headers.update(headers_extra)
    try:
        async with pool._client.stream("POST", path, json=body, headers=headers) as resp:
            if resp.status_code == 429:
                text = (await resp.aread()).decode(errors="replace")
                pool.note_rate_limited(acc)
                raise UpstreamError(429, text)
            if resp.status_code >= 500 or resp.status_code == 408:
                text = (await resp.aread()).decode(errors="replace")
                pool.note_error(acc, resp.status_code)
                raise UpstreamError(resp.status_code, text)
            if resp.status_code >= 400:
                text = (await resp.aread()).decode(errors="replace")
                # Client error (bad model, bad request) - not the account's
                # fault; do not fail over, surface immediately.
                raise UpstreamError(-resp.status_code, text)  # negative = terminal
            async for chunk in resp.aiter_bytes():
                yield chunk
    except httpx.HTTPError as e:
        pool.note_error(acc, 599)
        raise UpstreamError(599, f"connection error: {e}") from e


async def route_request(pool: AccountPool, path: str, body: dict,
                        stream: bool, auth_style: str = "bearer"):
    """Pick an account, try, fail over. Returns (account_name, response_or_gen).
    Raises UpstreamError if every account failed."""
    tried: set[str] = set()
    last_err: Optional[UpstreamError] = None
    while True:
        acc = pool.pick(exclude=tried)
        if acc is None:
            raise UpstreamError(503, f"all accounts exhausted; last error: {last_err}")
        tried.add(acc.name)
        pool.note_request(acc)
        try:
            if stream:
                gen = stream_attempt(pool, acc, path, body, auth_style=auth_style)
                # Consume the first chunk to surface pre-stream failures and
                # enable failover before anything reaches the client.
                first = None
                try:
                    first = await gen.__anext__()
                except StopAsyncIteration:
                    first = b""
                return acc.name, _prefixed(gen, first)
            else:
                resp = await attempt(pool, acc, path, body, auth_style=auth_style)
                return acc.name, resp
        except UpstreamError as e:
            if e.status < 0:  # terminal client error - no failover
                raise UpstreamError(-e.status, e.body) from e
            last_err = e
            log.info("failover from [%s]: %s", acc.name, e)


async def _prefixed(gen: AsyncIterator[bytes], first: bytes) -> AsyncIterator[bytes]:
    if first:
        yield first
    async for chunk in gen:
        yield chunk


# Fields the InkStone OpenAI endpoint is known to accept. Used as a fallback
# sanitizer when the upstream rejects a request with 400 (harnesses like
# ZCode send exotic fields - thinking/enable_thinking/reasoning/metadata/... -
# some of which the upstream can refuse).
_ALLOWED_CHAT_FIELDS = {
    "model", "messages", "frequency_penalty", "logit_bias", "logprobs",
    "top_logprobs", "max_tokens", "n", "presence_penalty", "response_format",
    "seed", "stop", "stream", "stream_options", "temperature", "top_p",
    "tools", "tool_choice", "user", "functions", "function_call",
    "max_completion_tokens",
}


def sanitize_body(body: dict) -> dict:
    clean = {k: v for k, v in body.items() if k in _ALLOWED_CHAT_FIELDS}
    msgs = []
    for m in clean.get("messages", []):
        if not isinstance(m, dict):
            continue
        m = dict(m)
        content = m.get("content")
        # Flatten typed content parts to plain text (upstream is strict).
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            m["content"] = "\n".join(p for p in parts if p)
        msgs.append(m)
    clean["messages"] = msgs
    return clean


def build_app(pool: AccountPool, proxy_key: str = ""):
    import os
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    debug_dump = os.environ.get("INKSTONE_PROXY_DEBUG", "") == "1"

    def _dump_request(body: dict, path: str) -> None:
        if not debug_dump:
            return
        try:
            with open("/tmp/inkstone_proxy_last_request.json", "w") as f:
                json.dump({"path": path, "body": body, "ts": time.time()}, f, indent=2)
        except Exception:
            pass

    def _log_upstream_error(path: str, status: int, body_text: str) -> None:
        log.warning("upstream %s -> %d: %s", path, status, body_text[:500])

    app = FastAPI(title="inkstone-proxy", docs_url=None, redoc_url=None)

    def _authorized(req: Request) -> bool:
        if not proxy_key:
            return True
        auth = req.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:] == proxy_key
        return req.headers.get("x-api-key") == proxy_key

    def _unauthorized() -> JSONResponse:
        return JSONResponse({"error": {"message": "invalid proxy key",
                                       "type": "auth_error"}}, status_code=401)

    @app.get("/health")
    async def health():
        return {"ok": True, "accounts": len(pool.accounts)}

    @app.get("/quota")
    async def quota():
        return pool.snapshot()

    @app.get("/v1/models")
    async def models():
        if not pool._models_fetched:
            await pool.refresh_models()
        return {"object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "inkstone"}
                         for m in pool.models]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if not _authorized(req):
            return _unauthorized()
        body = await req.json()
        _dump_request(body, "/v1/chat/completions")
        stream = bool(body.get("stream"))
        try:
            acc_name, result = await route_request(pool, OPENAI_CHAT, body, stream)
        except UpstreamError as e:
            if abs(e.status) == 400:
                # Retry once with a sanitized body - harnesses send fields the
                # upstream may refuse. Log the original rejection for diagnosis.
                log.warning("chat 400 - retrying sanitized. original: %s", e.body[:300])
                try:
                    acc_name, result = await route_request(
                        pool, OPENAI_CHAT, sanitize_body(body), stream)
                    log.warning("chat sanitized retry SUCCEEDED (culprit field stripped)")
                except UpstreamError as e2:
                    _log_upstream_error("/v1/chat/completions", abs(e2.status), e2.body)
                    return JSONResponse(
                        {"error": {"message": e2.body, "type": "upstream_error"}},
                        status_code=max(400, min(599, abs(e2.status))))
            else:
                _log_upstream_error("/v1/chat/completions", abs(e.status), e.body)
                return JSONResponse({"error": {"message": e.body, "type": "upstream_error"}},
                                    status_code=max(400, min(599, abs(e.status))))
        if stream:
            return StreamingResponse(result, media_type="text/event-stream",
                                     headers={"x-inkstone-account": acc_name})
        resp = result
        if resp.status_code >= 400:
            _log_upstream_error("/v1/chat/completions", resp.status_code, resp.text)
        try:
            data = resp.json()
            usage = data.get("usage") or {}
            acc = next(a for a in pool.accounts if a.name == acc_name)
            acc.record_usage(usage)
        except Exception:
            pass
        return JSONResponse(json.loads(resp.text), status_code=resp.status_code,
                            headers={"x-inkstone-account": acc_name})

    @app.post("/v1/messages")
    async def messages(req: Request):
        if not _authorized(req):
            return _unauthorized()
        body = await req.json()
        stream = bool(body.get("stream"))
        try:
            acc_name, result = await route_request(pool, ANTHROPIC_MESSAGES, body,
                                                   stream, auth_style="anthropic")
        except UpstreamError as e:
            return JSONResponse({"type": "error",
                                 "error": {"type": "upstream_error", "message": e.body}},
                                status_code=max(400, min(599, abs(e.status))))
        if stream:
            return StreamingResponse(result, media_type="text/event-stream",
                                     headers={"x-inkstone-account": acc_name})
        resp = result
        try:
            data = resp.json()
            usage = data.get("usage") or {}
            acc = next(a for a in pool.accounts if a.name == acc_name)
            acc.record_usage({"prompt_tokens": usage.get("input_tokens", 0),
                              "completion_tokens": usage.get("output_tokens", 0)})
        except Exception:
            pass
        return JSONResponse(json.loads(resp.text), status_code=resp.status_code,
                            headers={"x-inkstone-account": acc_name})

    @app.on_event("startup")
    async def _startup():
        pool.start()

    return app


def load_pool(config_path: str, only: list[str] | None = None) -> AccountPool:
    with open(config_path) as f:
        cfg = json.load(f)
    accounts = []
    for name, acc in cfg["accounts"].items():
        if only and name not in only:
            continue
        if not acc.get("api_key"):
            log.warning("[%s] no api_key - skipped", name)
            continue
        accounts.append(AccountState(
            name=name, api_key=acc["api_key"],
            uaa_token=acc.get("uaa_token", ""), ssouid=str(acc.get("ssouid", ""))))
    if not accounts:
        raise SystemExit("no accounts with api_key found in config")
    return AccountPool(accounts)


def serve(host: str = "127.0.0.1", port: int = 8787, proxy_key: str = "",
          accounts: list[str] | None = None, config_path: str = "config.json"):
    import uvicorn
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    pool = load_pool(config_path, accounts)
    app = build_app(pool, proxy_key)
    names = ", ".join(a.name for a in pool.accounts)
    log.info("inkstone-proxy on http://%s:%d | accounts: %s%s",
             host, port, names, " | proxy key: ON" if proxy_key else "")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
