from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from bot.config import BotConfig, ConfigError
from bot.data_store import ModerationStore
from bot.moderation_bot import ModerationBot
from bot.telegram_api import TelegramAPI


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        val = value.strip()
        # remove inline comments for unquoted values
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            # strip surrounding quotes only
            val = val[1:-1]
        else:
            if "#" in val:
                val = val.split("#", 1)[0].rstrip()
        os.environ.setdefault(key, val)


def build_logger() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    load_env_file()
    build_logger()
    try:
        config = BotConfig.from_env()
    except ConfigError as exc:
        logging.getLogger("bottgmoder").error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    store = ModerationStore(config.storage_path)
    api = TelegramAPI(config.token)
    bot = ModerationBot(config, store, api)
    bot.run_forever()


if __name__ == "__main__":
    main()
