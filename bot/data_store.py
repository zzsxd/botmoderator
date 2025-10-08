from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional


class ModerationStore:
    """Thread-safe persistent storage for moderation state."""

    def __init__(self, storage_path: str) -> None:
        self._path = Path(storage_path)
        self._lock = threading.RLock()
        self._data = {"moderated_chats": {}, "global_keywords": []}
        self._ensure_directory()
        self._load()

    def _ensure_directory(self) -> None:
        directory = self._path.parent
        if directory and not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                self._data = json.load(fh)
        except json.JSONDecodeError:
            # Keep defaults if file is corrupted.
            pass
        # Ensure required keys exist in loaded data
        self._data.setdefault("moderated_chats", {})
        self._data.setdefault("global_keywords", [])

    def _persist(self) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(self._path)

    def _get_chat_entry(self, chat_id: int, create: bool = False) -> Optional[Dict[str, object]]:
        key = str(chat_id)
        chats = self._data.setdefault("moderated_chats", {})
        if create:
            return chats.setdefault(
                key,
                {
                    "title": "",
                    "keywords": [],
                    "warnings": {},
                },
            )
        return chats.get(key)

    def add_chat(self, chat_id: int, title: Optional[str] = None) -> bool:
        key = str(chat_id)
        with self._lock:
            chats = self._data.setdefault("moderated_chats", {})
            exists = key in chats
            entry = chats.setdefault(
                key,
                {
                    "title": "",
                    "keywords": [],
                    "warnings": {},
                },
            )
            if title is not None:
                entry["title"] = title
            self._persist()
            return not exists

    def update_chat_title(self, chat_id: int, title: str) -> None:
        with self._lock:
            entry = self._get_chat_entry(chat_id, create=True)
            entry["title"] = title
            self._persist()

    def remove_chat(self, chat_id: int) -> bool:
        key = str(chat_id)
        with self._lock:
            removed = self._data["moderated_chats"].pop(key, None) is not None
            if removed:
                self._persist()
            return removed

    def list_chats(self) -> Dict[int, Dict[str, object]]:
        with self._lock:
            result: Dict[int, Dict[str, object]] = {}
            global_keywords = list(self._data.get("global_keywords", []))
            for chat_id_str, payload in self._data["moderated_chats"].items():
                result[int(chat_id_str)] = {
                    "title": payload.get("title", ""),
                    # Report current global keywords for every chat (unified list)
                    "keywords": list(global_keywords),
                    "warnings": {int(uid): cnt for uid, cnt in payload.get("warnings", {}).items()},
                }
            return result

    def add_keyword(self, chat_id: int, keyword: str) -> bool:
        # Backward compatibility: add to global list, ignore chat_id
        return self.add_keywords([keyword]) > 0

    def add_keywords(self, words: List[str]) -> int:
        cleaned = [w.strip() for w in words if w and w.strip()]
        if not cleaned:
            return 0
        with self._lock:
            current: List[str] = self._data.setdefault("global_keywords", [])
            existing_cf = {w.casefold() for w in current}
            added = [w for w in cleaned if w.casefold() not in existing_cf]
            if not added:
                return 0
            current.extend(added)
            self._persist()
            return len(added)

    def remove_keyword(self, chat_id: int, keyword: str) -> bool:
        return self.remove_keywords([keyword]) > 0

    def remove_keywords(self, words: List[str]) -> int:
        targets = [w.strip() for w in words if w and w.strip()]
        if not targets:
            return 0
        with self._lock:
            current: List[str] = self._data.setdefault("global_keywords", [])
            to_remove_cf = {w.casefold() for w in targets}
            before = len(current)
            current[:] = [w for w in current if w.casefold() not in to_remove_cf]
            removed = before - len(current)
            if removed:
                self._persist()
            return removed

    def get_keywords(self, chat_id: int) -> List[str]:
        # Backward compatibility: return global keywords for any chat
        with self._lock:
            return list(self._data.get("global_keywords", []))

    def list_global_keywords(self) -> List[str]:
        with self._lock:
            return list(self._data.get("global_keywords", []))

    def is_chat_moderated(self, chat_id: int) -> bool:
        with self._lock:
            return str(chat_id) in self._data.get("moderated_chats", {})

    def increment_warning(self, chat_id: int, user_id: int) -> int:
        with self._lock:
            entry = self._get_chat_entry(chat_id, create=True)
            warnings: Dict[str, int] = entry.setdefault("warnings", {})
            user_key = str(user_id)
            warnings[user_key] = warnings.get(user_key, 0) + 1
            self._persist()
            return warnings[user_key]

    def reset_warnings(self, chat_id: int, user_id: int) -> bool:
        with self._lock:
            entry = self._get_chat_entry(chat_id)
            if not entry:
                return False
            warnings: Dict[str, int] = entry.get("warnings", {})
            if str(user_id) not in warnings:
                return False
            warnings.pop(str(user_id), None)
            self._persist()
            return True

    def get_warning(self, chat_id: int, user_id: int) -> int:
        with self._lock:
            entry = self._get_chat_entry(chat_id)
            if not entry:
                return 0
            warnings: Dict[str, int] = entry.get("warnings", {})
            return int(warnings.get(str(user_id), 0))

    def get_all_warnings(self, chat_id: int) -> Dict[int, int]:
        with self._lock:
            entry = self._get_chat_entry(chat_id)
            if not entry:
                return {}
            return {int(uid): cnt for uid, cnt in entry.get("warnings", {}).items()}


__all__ = ["ModerationStore"]
