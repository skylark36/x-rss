"""统一的 LLM 客户端。

面向任意 OpenAI 兼容的 `/chat/completions` 服务，只依赖 requests。
设计目标是让上层模块（如 filters.py）只关心提示词与业务判定，不必重复处理
密钥读取、重试、超时、JSON 容错与用量统计。

用法::

    from llm import LLMClient

    client = LLMClient.from_env()          # 读取 LLM_* 环境变量
    if client.available:
        text = client.chat([{"role": "user", "content": "你好"}])
        data = client.json([{"role": "user", "content": "返回 JSON: {\"a\":1}"}])

命令行冒烟测试::

    python llm.py "用一句话解释什么是资产配置"
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 可重试的 HTTP 状态码
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------- 异常


class LLMError(Exception):
    """所有 LLM 相关异常的基类。"""


class LLMUnavailable(LLMError):
    """未配置密钥，或服务明确不可用（如 401/403），重试无意义。"""


class LLMRequestError(LLMError):
    """重试次数用尽后仍然失败。"""


class LLMParseError(LLMError):
    """响应内容无法解析为期望的结构。"""


# ---------------------------------------------------------------- 配置


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    # 主模型失败时依次尝试的备用模型（同一服务内）
    fallback_models: List[str] = field(default_factory=list)

    timeout: int = 60
    retries: int = 3
    # 指数退避的基准秒数，实际等待 = base * 2**(attempt-1) + 抖动
    retry_base_delay: float = 1.0
    retry_max_delay: float = 20.0

    temperature: float = 0.0
    max_tokens: int = 2048

    @classmethod
    def from_env(cls, prefix: str = "LLM", **overrides) -> "LLMConfig":
        """从 <PREFIX>_API_KEY / _BASE_URL / _MODEL / _FALLBACK_MODELS 读取。

        overrides 中的显式参数优先于环境变量，常用于测试或局部覆盖。
        """
        fallback_raw = os.getenv(f"{prefix}_FALLBACK_MODELS") or ""
        cfg = cls(
            api_key=os.getenv(f"{prefix}_API_KEY"),
            base_url=(os.getenv(f"{prefix}_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            model=os.getenv(f"{prefix}_MODEL") or DEFAULT_MODEL,
            fallback_models=[m.strip() for m in fallback_raw.split(",") if m.strip()],
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.base_url = (cfg.base_url or DEFAULT_BASE_URL).rstrip("/")
        return cfg

    @classmethod
    def from_dict(cls, data: Optional[dict], prefix: str = "LLM", **overrides) -> "LLMConfig":
        """从配置字典的 [llm] 段落读取，环境变量优先于字典默认值。

        字典里 api_key_env / base_url_env / model_env 指定要读取的环境变量名，
        default_base_url / default_model 作为环境变量缺失时的兜底。
        """
        data = data or {}
        base = cls(
            api_key=os.getenv(data.get("api_key_env", f"{prefix}_API_KEY")),
            base_url=(
                os.getenv(data.get("base_url_env", f"{prefix}_BASE_URL"))
                or data.get("default_base_url")
                or DEFAULT_BASE_URL
            ).rstrip("/"),
            model=(
                os.getenv(data.get("model_env", f"{prefix}_MODEL"))
                or data.get("default_model")
                or DEFAULT_MODEL
            ),
            fallback_models=list(data.get("fallback_models") or []),
            timeout=int(data.get("timeout", 60)),
            retries=int(data.get("retries", 3)),
            max_tokens=int(data.get("max_tokens", 2048)),
            temperature=float(data.get("temperature", 0.0)),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        return base


# ---------------------------------------------------------------- 用量统计


@dataclass
class Usage:
    requests: int = 0
    failed_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0):
        self.requests += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __str__(self) -> str:
        return (
            f"{self.requests} req ({self.failed_requests} failed), "
            f"{self.prompt_tokens} prompt + {self.completion_tokens} completion "
            f"= {self.total_tokens} tokens"
        )


# ---------------------------------------------------------------- 客户端


class LLMClient:
    """OpenAI 兼容 chat 服务的轻量封装。

    - chat()：返回纯文本
    - json()：返回已解析的 JSON（自动剥离 ``` 包裹、截取最外层括号）
    - batch_json()：并发分批处理长列表，单批失败不影响其他批次
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.usage = Usage()
        self._local = threading.local()
        self._lock = threading.Lock()
    # -- 构造快捷方式 ------------------------------------------------

    @classmethod
    def from_env(cls, prefix: str = "LLM", **overrides) -> "LLMClient":
        return cls(LLMConfig.from_env(prefix, **overrides))

    @classmethod
    def from_dict(cls, data: Optional[dict], prefix: str = "LLM", **overrides) -> "LLMClient":
        return cls(LLMConfig.from_dict(data, prefix, **overrides))

    # -- 能力探测 ----------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    @property
    def model(self) -> str:
        return self.config.model

    def __repr__(self) -> str:
        state = "ready" if self.available else "no-api-key"
        return f"<LLMClient {state} model={self.config.model} base={self.config.base_url}>"

    # -- 会话 --------------------------------------------------------

    @property
    def _session(self) -> requests.Session:
        """每个线程独立 Session，避免连接池跨线程竞争。"""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    # -- 底层请求 ----------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> str:
        """发起一次对话请求并返回 assistant 文本。

        失败会按指数退避重试；全部模型与重试都用尽后抛 LLMRequestError。
        """
        if not self.available:
            raise LLMUnavailable(
                f"No API key configured (looked for {self.config.base_url}). "
                "Set LLM_API_KEY or construct LLMConfig explicitly."
            )

        models = [model or self.config.model] + [
            m for m in self.config.fallback_models if m != (model or self.config.model)
        ]
        last_error: Optional[Exception] = None

        for candidate in models:
            try:
                return self._chat_once(
                    messages,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=extra,
                )
            except LLMUnavailable:
                raise
            except LLMRequestError as e:
                last_error = e
                if candidate != models[-1]:
                    logger.warning(f"Model {candidate} exhausted retries, trying fallback model.")
                continue

        raise last_error or LLMRequestError("LLM request failed for unknown reason")

    def _chat_once(
        self,
        messages: Sequence[dict],
        *,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra: dict,
    ) -> str:
        url = f"{self.config.base_url}/chat/completions"
        body: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        body.update(extra)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        while attempt < self.config.retries:
            attempt += 1
            try:
                resp = self._session.post(
                    url, headers=headers, json=body, timeout=self.config.timeout
                )
            except Exception as e:  # 网络层错误
                logger.warning(
                    f"LLM request attempt {attempt}/{self.config.retries} failed: "
                    f"{type(e).__name__}: {e}"
                )
                self._sleep_before_retry(attempt)
                continue

            if resp.status_code == 200:
                return self._extract_content(resp, model)

            # 认证类错误：重试没有意义
            if resp.status_code in (401, 403):
                raise LLMUnavailable(
                    f"LLM auth failed (HTTP {resp.status_code}): {resp.text[:200]}"
                )

            if resp.status_code not in RETRYABLE_STATUS:
                raise LLMRequestError(
                    f"LLM HTTP {resp.status_code} (non-retryable): {resp.text[:200]}"
                )

            logger.warning(
                f"LLM request attempt {attempt}/{self.config.retries} got HTTP "
                f"{resp.status_code}: {resp.text[:160]}"
            )
            self._sleep_before_retry(attempt, resp)

        with self._lock:
            self.usage.failed_requests += 1
        raise LLMRequestError(
            f"LLM request to {self.config.base_url} failed after "
            f"{self.config.retries} attempts (model={model})"
        )

    def _extract_content(self, resp: requests.Response, model: str) -> str:
        try:
            data = resp.json()
        except ValueError as e:
            raise LLMRequestError(f"LLM returned non-JSON body: {resp.text[:200]}") from e

        with self._lock:
            self.usage.add(
                int((data.get("usage") or {}).get("prompt_tokens") or 0),
                int((data.get("usage") or {}).get("completion_tokens") or 0),
            )

        choices = data.get("choices") or []
        if not choices:
            raise LLMRequestError(f"LLM response has no choices: {str(data)[:200]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            # 部分推理模型把结果放在 reasoning_content
            content = message.get("reasoning_content")
        if not isinstance(content, str):
            raise LLMRequestError(f"LLM response content is not text: {str(message)[:200]}")
        return content

    def _sleep_before_retry(self, attempt: int, resp: Optional[requests.Response] = None):
        delay = min(
            self.config.retry_base_delay * (2 ** (attempt - 1)),
            self.config.retry_max_delay,
        )
        delay += random.uniform(0, 0.5)
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
        # 最后一次失败后不再等待
        if attempt < self.config.retries:
            time.sleep(delay)

    # -- JSON 支持 ---------------------------------------------------

    @staticmethod
    def _strip_to_json(text: str) -> str:
        """剥掉 markdown 代码围栏，截取最外层 JSON 值。"""
        text = text.strip()
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # 优先找最外层对象，其次数组
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start, end = text.find(open_ch), text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                return text[start : end + 1]
        return text

    def json(
        self,
        messages: Sequence[dict],
        *,
        expect: Optional[type] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        retry_on_parse_error: bool = True,
        **kwargs: Any,
    ) -> Any:
        """请求并解析 JSON。

        expect: 期望的顶层类型（dict / list），不符时抛 LLMParseError。
        validator: 额外的结构校验，返回 False 视为解析失败。
        retry_on_parse_error: 解析失败时追加一次「只输出 JSON」的纠正请求。
        """
        raw = self.chat(messages, **kwargs)
        attempts = 2 if retry_on_parse_error else 1
        last_error: Optional[Exception] = None

        for i in range(attempts):
            try:
                parsed = self._parse_json(raw, expect, validator)
                return parsed
            except LLMParseError as e:
                last_error = e
                if i + 1 >= attempts:
                    break
                logger.warning(f"JSON parse failed ({e}); asking model to correct output.")
                raw = self.chat(
                    list(messages)
                    + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": "上面的输出无法被解析为合法 JSON。"
                            "请只输出合法 JSON，不要任何解释文字，不要 markdown 代码块。",
                        },
                    ],
                    **kwargs,
                )

        raise last_error or LLMParseError("failed to parse LLM JSON response")

    @classmethod
    def _parse_json(
        cls,
        raw: str,
        expect: Optional[type] = None,
        validator: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        if not raw or not raw.strip():
            raise LLMParseError("empty response")
        candidate = cls._strip_to_json(raw)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise LLMParseError(f"invalid JSON: {e}; raw={raw[:200]}") from e
        if expect is not None and not isinstance(parsed, expect):
            raise LLMParseError(
                f"expected {expect.__name__}, got {type(parsed).__name__}: {str(parsed)[:200]}"
            )
        if validator is not None and not validator(parsed):
            raise LLMParseError(f"validation failed: {str(parsed)[:200]}")
        return parsed

    # -- 批量处理 ----------------------------------------------------

    def batch_json(
        self,
        items: Sequence[Any],
        build_messages: Callable[[Sequence[Any]], Sequence[dict]],
        *,
        batch_size: int = 40,
        max_workers: int = 1,
        expect: Optional[type] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        **kwargs: Any,
    ) -> List[Optional[Any]]:
        """把 items 切批并发请求，返回与 items 等长、逐项对齐的结果列表。

        build_messages(batch) 负责生成该批次的 messages（通常带上 index）。
        单批失败时该批所有位置填 None，由调用方决定兜底策略。
        """
        items = list(items)
        if not items:
            return []

        chunks = [
            (start, items[start : start + batch_size])
            for start in range(0, len(items), batch_size)
        ]

        def run(chunk) -> tuple:
            start, batch = chunk
            try:
                return start, self.json(
                    build_messages(batch),
                    expect=expect,
                    validator=validator,
                    **kwargs,
                )
            except LLMError as e:
                logger.error(f"LLM batch starting at {start} failed: {e}")
                return start, None
            except Exception as e:  # 非 LLMError 也不应拖垮整批调用方
                logger.error(
                    f"LLM batch starting at {start} crashed: {type(e).__name__}: {e}"
                )
                return start, None

        results: List[Optional[Any]] = [None] * len(items)
        if max_workers <= 1:
            outcomes = [run(c) for c in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as pool:
                outcomes = list(pool.map(run, chunks))

        for start, payload in outcomes:
            if payload is not None:
                results[start] = payload
        return results

    # -- 资源 --------------------------------------------------------

    def close(self):
        session = getattr(self._local, "session", None)
        if session is not None:
            session.close()
            self._local.session = None

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info):
        self.close()


# ---------------------------------------------------------------- CLI


def _main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LLM client smoke test")
    parser.add_argument("prompt", nargs="?", default="用一句话解释什么是资产配置。")
    parser.add_argument("--json", action="store_true", help="要求模型返回 JSON 并解析")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--base-url", help="覆盖服务地址")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    client = LLMClient.from_env(**overrides)

    print(client)
    if not client.available:
        print("未配置 LLM_API_KEY，无法测试。", file=os.sys.stderr)
        return 2

    messages = [{"role": "user", "content": args.prompt}]
    try:
        if args.json:
            print(json.dumps(client.json(messages), ensure_ascii=False, indent=2))
        else:
            print(client.chat(messages))
    finally:
        print("usage:", client.usage)
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(os.sys.argv[1:]))
