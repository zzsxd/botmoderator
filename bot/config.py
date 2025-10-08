from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Set


class ConfigError(RuntimeError):
    """Raised when the configuration is invalid."""


def _parse_int_set(raw: Optional[str]) -> Set[int]:
    if not raw:
        return set()
    values: Set[int] = set()
    for item in raw.split(","):
        trimmed = item.strip()
        # allow inline comments in comma-separated lists
        if "#" in trimmed:
            trimmed = trimmed.split("#", 1)[0].strip()
        if not trimmed:
            continue
        try:
            values.add(int(trimmed))
        except ValueError as exc:  # pragma: no cover - configuration validation
            raise ConfigError(f"Cannot parse integer value from '{trimmed}'") from exc
    return values


def _get_int_env(name: str, default: int) -> int:
    """Read integer env var and ignore trailing comments/extra whitespace."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    # If spaces remain, take the first token (e.g., "3   some")
    if " " in value or "\t" in value:
        value = value.split()[0]
    return int(value)


@dataclass(frozen=True)
class BotConfig:
    token: str
    admin_chat_ids: Set[int] = field(default_factory=set)
    admin_user_ids: Set[int] = field(default_factory=set)
    warning_limit: int = 3
    long_poll_timeout: int = 25
    worker_pool_size: int = 4
    storage_path: str = "moderation_state.json"

    @staticmethod
    def from_env(prefix: str = "BOT_") -> "BotConfig":
        token = os.getenv(f"{prefix}TOKEN", "").strip()
        if not token:
            raise ConfigError("Environment variable BOT_TOKEN must be set")

        admin_chat_ids = _parse_int_set(os.getenv(f"{prefix}ADMIN_CHAT_IDS"))
        admin_user_ids = _parse_int_set(os.getenv(f"{prefix}ADMIN_USER_IDS"))

        if not admin_chat_ids and not admin_user_ids:
            raise ConfigError(
                "Provide at least one admin chat id (BOT_ADMIN_CHAT_IDS) or admin user id (BOT_ADMIN_USER_IDS)"
            )

        warning_limit = _get_int_env(f"{prefix}WARNING_LIMIT", 3)
        long_poll_timeout = _get_int_env(f"{prefix}LONG_POLL_TIMEOUT", 25)
        worker_pool_size = _get_int_env(f"{prefix}WORKER_POOL_SIZE", 6)
        storage_path = os.getenv(f"{prefix}STORAGE_PATH", "moderation_state.json").strip()

        return BotConfig(
            token=token,
            admin_chat_ids=admin_chat_ids,
            admin_user_ids=admin_user_ids,
            warning_limit=warning_limit,
            long_poll_timeout=long_poll_timeout,
            worker_pool_size=worker_pool_size,
            storage_path=storage_path,
        )


__all__ = ["BotConfig", "ConfigError"]
