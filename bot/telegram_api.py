from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests


logger = logging.getLogger(__name__)


class TelegramAPIError(RuntimeError):
    """Raised when the Telegram Bot API returns an error."""


class TelegramAPI:
    """Minimal wrapper around the Telegram Bot API using long polling."""

    def __init__(self, token: str, session: Optional[requests.Session] = None) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._session = session or requests.Session()

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self._base_url}/{method}"
        try:
            response = self._session.post(url, json=params or {}, timeout=60)
        except requests.RequestException as exc:  # pragma: no cover - depends on network I/O
            logger.error("Request to Telegram API failed: %s", exc)
            raise TelegramAPIError(str(exc)) from exc

        if response.status_code != 200:
            logger.error("Telegram API returned HTTP %s: %s", response.status_code, response.text)
            raise TelegramAPIError(f"HTTP {response.status_code}: {response.text}")

        payload = response.json()
        if not payload.get("ok"):
            description = payload.get("description", "Unknown error")
            logger.error("Telegram API method %s failed: %s", method, description)
            raise TelegramAPIError(description)
        return payload.get("result")

    def get_updates(self, offset: Optional[int], timeout: int) -> Any:
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        return self.call("getUpdates", params)

    def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        return self.call("sendMessage", params)

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params: Dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        return self.call("editMessageText", params)

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> Any:
        params: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text
        if show_alert:
            params["show_alert"] = True
        return self.call("answerCallbackQuery", params)

    def delete_message(self, chat_id: int, message_id: int) -> Any:
        return self.call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    def ban_chat_member(self, chat_id: int, user_id: int, until_date: Optional[int] = None) -> Any:
        params: Dict[str, Any] = {"chat_id": chat_id, "user_id": user_id}
        if until_date is not None:
            params["until_date"] = until_date
        return self.call("banChatMember", params)

    def unban_chat_member(self, chat_id: int, user_id: int) -> Any:
        return self.call("unbanChatMember", {"chat_id": chat_id, "user_id": user_id})

    def get_chat_administrators(self, chat_id: int) -> Any:
        return self.call("getChatAdministrators", {"chat_id": chat_id})

    def get_chat(self, chat_id: int) -> Any:
        return self.call("getChat", {"chat_id": chat_id})


__all__ = ["TelegramAPI", "TelegramAPIError"]
