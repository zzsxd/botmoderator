from __future__ import annotations

import logging
import shlex
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple, Deque
from collections import deque, defaultdict

from .config import BotConfig
from .data_store import ModerationStore
from .telegram_api import TelegramAPI, TelegramAPIError


logger = logging.getLogger(__name__)


class ModerationBot:
    """Implements the moderation workflow and admin commands."""

    def __init__(self, config: BotConfig, store: ModerationStore, api: TelegramAPI) -> None:
        self.config = config
        self.store = store
        self.api = api
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=config.worker_pool_size, thread_name_prefix="worker")
        self._polling_thread = threading.Thread(target=self._poll_updates, name="polling-thread", daemon=True)
        self._offset: Optional[int] = None
        # Admin conversation sessions per admin chat
        self._admin_sessions_lock = threading.RLock()
        self._admin_sessions: Dict[int, Dict[str, Any]] = {}
        # UI message ids for inline menu per admin chat (legacy inline menu)
        self._admin_ui_lock = threading.RLock()
        self._admin_menu_message: Dict[int, int] = {}

        # Message length threshold for auto-moderation
        self.MAX_MESSAGE_LENGTH = 100

        # Anti-flood settings (configurable via env): default 10 msgs per 60 seconds
        self.RL_WINDOW_SECONDS = getattr(config, "rl_window_seconds", 60)
        self.RL_MAX_MESSAGES = getattr(config, "rl_max_messages", 10)
        # chat_id -> user_id -> deque[timestamps]
        self._rate_buckets: Dict[int, Dict[int, Deque[float]]] = defaultdict(lambda: defaultdict(deque))
        self._rate_lock = threading.RLock()

        # Button labels (RU)
        self.BTN_MENU = "Меню"
        # Root groups (reply keyboard)
        self.BTN_GROUP_CHATS = "Чаты"
        self.BTN_GROUP_WORDS = "Слова"
        self.BTN_BACK = "⬅ Назад"
        # Chats submenu
        self.BTN_LIST_CHATS = "Модерируемые чаты"
        self.BTN_ADD_CHAT = "Добавить чат"
        self.BTN_REMOVE_CHAT = "Удалить чат"
        # Words submenu
        self.BTN_ADD_WORD = "Добавить слово"
        self.BTN_REMOVE_WORD = "Удалить слово"
        self.BTN_LIST_WORDS = "Список слов"
        # Other
        self.BTN_WARNINGS = "Предупреждения"
        self.BTN_RESET_WARNING = "Сброс предупреждений"
        self.BTN_HELP = "Помощь"
        self.BTN_HIDE_KB = "Скрыть клавиатуру"

        # Request IDs for reply keyboard 'request_chat' buttons
        self.REQ_ADD_CHAT = 101
        self.REQ_REMOVE_CHAT = 102

    def start(self) -> None:
        logger.info("Starting moderation bot")
        self._polling_thread.start()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping bot...")
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5)
        logger.info("Moderation bot stopped")

    def _poll_updates(self) -> None:
        logger.info("Polling thread started")
        while not self._stop_event.is_set():
            try:
                updates = self.api.get_updates(self._offset, self.config.long_poll_timeout)
            except TelegramAPIError as exc:
                logger.warning("Failed to fetch updates: %s", exc)
                time.sleep(5)
                continue
            except Exception:  # pragma: no cover - network failure guard
                logger.exception("Unexpected error while fetching updates")
                time.sleep(5)
                continue

            if not updates:
                continue

            for update in updates:
                self._offset = update.get("update_id", 0) + 1
                self._executor.submit(self._handle_update, update)

    def _handle_update(self, update: Dict[str, Any]) -> None:
        # Handle callback queries first (inline keyboard)
        callback = update.get("callback_query")
        if callback:
            self._handle_callback(callback)
            return

        message: Optional[Dict[str, Any]] = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
            or update.get("edited_channel_post")
        )
        if not message:
            return

        if self._is_admin_context(message):
            self._handle_admin_message(message)
            return

        self._handle_moderation(message)

    def _is_admin_context(self, message: Dict[str, Any]) -> bool:
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        user = message.get("from", {})
        user_id = user.get("id")
        if chat_id in self.config.admin_chat_ids:
            return True
        if user_id in self.config.admin_user_ids and chat.get("type") == "private":
            return True
        return False

    def _handle_admin_message(self, message: Dict[str, Any]) -> None:
        text = message.get("text") or message.get("caption") or ""
        msg_id = message.get("message_id")
        # Handle chat selection share (KeyboardButtonRequestChat)
        chat_shared = message.get("chat_shared")
        if chat_shared:
            admin_chat_id = message["chat"]["id"]
            shared_chat_id = chat_shared.get("chat_id")
            request_id = chat_shared.get("request_id")
            # Primary flow: static request ids from main reply keyboard
            if request_id == self.REQ_ADD_CHAT:
                created = self.store.add_chat(shared_chat_id)
                # Try to capture chat title for nicer listings
                try:
                    info = self.api.get_chat(shared_chat_id)
                    title = info.get("title")
                    if title:
                        self.store.update_chat_title(shared_chat_id, title)
                except TelegramAPIError:
                    pass
                msg = "добавлен" if created else "уже в списке"
                self._send_to_chat(admin_chat_id, f"Чат {shared_chat_id} {msg}.", reply_markup=self._reply_keyboard("chats"))
                return
            if request_id == self.REQ_REMOVE_CHAT:
                ok = self.store.remove_chat(shared_chat_id)
                msg = "удален" if ok else "не найден"
                self._send_to_chat(admin_chat_id, f"Чат {shared_chat_id} {msg}.", reply_markup=self._reply_keyboard("chats"))
                return
            # Backward-compatible flow for older dynamic keyboards
            state = self._get_session(admin_chat_id)
            if state == "AWAIT_ADD_CHAT_CHOOSE":
                sess = self._get_session_obj(admin_chat_id)
                if sess and sess.get("data", {}).get("req_id") == request_id:
                    created = self.store.add_chat(shared_chat_id)
                    try:
                        info = self.api.get_chat(shared_chat_id)
                        title = info.get("title")
                        if title:
                            self.store.update_chat_title(shared_chat_id, title)
                    except TelegramAPIError:
                        pass
                    msg = "добавлен" if created else "уже в списке"
                    self._send_to_chat(admin_chat_id, f"Чат {shared_chat_id} {msg}.", reply_markup=self._reply_keyboard("chats"))
                    self._clear_session(admin_chat_id)
                    return
            if state == "AWAIT_REMOVE_CHAT_CHOOSE":
                sess = self._get_session_obj(admin_chat_id)
                if sess and sess.get("data", {}).get("req_id") == request_id:
                    ok = self.store.remove_chat(shared_chat_id)
                    msg = "удален" if ok else "не найден"
                    self._send_to_chat(admin_chat_id, f"Чат {shared_chat_id} {msg}.", reply_markup=self._reply_keyboard("chats"))
                    self._clear_session(admin_chat_id)
                    return

        try:
            tokens = shlex.split(text)
        except ValueError as exc:
            logger.warning("Failed to parse admin command: %s", exc)
            return
        command = tokens[0].lower() if tokens else ""
        args = tokens[1:] if tokens else []
        chat_id = message["chat"]["id"]

        # 1) Handle button presses (preferred UX)
        if text == self.BTN_MENU or text in ("/start", "/help", self.BTN_HELP):
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._clear_session(chat_id)
            self._show_root_menu(chat_id)
            return
        if text == self.BTN_BACK:
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._clear_session(chat_id)
            self._show_root_menu(chat_id)
            return
        if text == self.BTN_GROUP_CHATS:
            # Non-action label row
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._show_chats_menu(chat_id)
            return
        if text == self.BTN_LIST_CHATS:
            # Show list without echoing button text
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._cmd_list_chats(chat_id, [])
            return
        if text == self.BTN_GROUP_WORDS:
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._show_words_menu(chat_id)
            return
        if text == self.BTN_ADD_WORD:
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._set_session(chat_id, "AWAIT_ADD_KEYWORD")
            self._send_to_chat(chat_id, "Отправьте слова через запятую", reply_markup=self._reply_keyboard("words"))
            return
        if text == self.BTN_REMOVE_WORD:
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._set_session(chat_id, "AWAIT_REMOVE_KEYWORD")
            self._send_to_chat(chat_id, "Отправьте слова через запятую для удаления", reply_markup=self._reply_keyboard("words"))
            return
        if text == self.BTN_LIST_WORDS:
            self._maybe_delete_ui_echo(chat_id, msg_id, text)
            self._show_global_keywords(chat_id)
            return

        # 2) Handle active session step
        state = self._get_session(chat_id)
        if state:
            try:
                if state == "AWAIT_ADD_CHAT":
                    if not tokens:
                        raise ValueError("Нужно передать chat_id [описание]")
                    target_chat_id = self._parse_int(tokens[0], "chat_id")
                    title = " ".join(tokens[1:]) if len(tokens) > 1 else None
                    created = self.store.add_chat(target_chat_id, title)
                    if created:
                        self._send_to_chat(chat_id, f"Чат {target_chat_id} добавлен.", reply_markup=self._reply_keyboard("chats"))
                    else:
                        self._send_to_chat(chat_id, f"Чат {target_chat_id} уже есть. Данные обновлены.", reply_markup=self._reply_keyboard("chats"))
                elif state == "AWAIT_REMOVE_CHAT":
                    if not tokens:
                        raise ValueError("Нужно передать chat_id")
                    target_chat_id = self._parse_int(tokens[0], "chat_id")
                    ok = self.store.remove_chat(target_chat_id)
                    msg = "Удален." if ok else "Не найден."
                    self._send_to_chat(chat_id, f"Чат {target_chat_id}: {msg}", reply_markup=self._reply_keyboard("chats"))
                elif state == "AWAIT_ADD_KEYWORD":
                    words = self._parse_words_csv(text)
                    if not words:
                        raise ValueError("Нужно отправить слова через запятую")
                    added = self.store.add_keywords(words)
                    if added:
                        self._send_to_chat(chat_id, f"Добавлено слов: {added}.", reply_markup=self._reply_keyboard("words"))
                    else:
                        self._send_to_chat(chat_id, "Новые слова не обнаружены.", reply_markup=self._reply_keyboard("words"))
                elif state == "AWAIT_REMOVE_KEYWORD":
                    words = self._parse_words_csv(text)
                    if not words:
                        raise ValueError("Нужно отправить слова через запятую")
                    removed = self.store.remove_keywords(words)
                    if removed:
                        self._send_to_chat(chat_id, f"Удалено слов: {removed}.", reply_markup=self._reply_keyboard("words"))
                    else:
                        self._send_to_chat(chat_id, "Совпадений для удаления не найдено.", reply_markup=self._reply_keyboard("words"))
                elif state == "AWAIT_WARNINGS":
                    if not tokens:
                        raise ValueError("Нужно передать chat_id [user_id]")
                    target_chat_id = self._parse_int(tokens[0], "chat_id")
                    if len(tokens) == 1:
                        warnings = self.store.get_all_warnings(target_chat_id)
                        if not warnings:
                            self._send_to_chat(chat_id, f"Для чата {target_chat_id} нет предупреждений.", reply_markup=self._reply_keyboard("root"))
                        else:
                            lines = [f"{uid}: {cnt}" for uid, cnt in warnings.items()]
                            self._send_to_chat(chat_id, f"Предупреждения в чате {target_chat_id}:\n" + "\n".join(lines), reply_markup=self._reply_keyboard("root"))
                    else:
                        user_id = self._parse_int(tokens[1], "user_id")
                        cnt = self.store.get_warning(target_chat_id, user_id)
                        self._send_to_chat(chat_id, f"Пользователь {user_id}: {cnt} предупреждений в {target_chat_id}.", reply_markup=self._reply_keyboard("root"))
                elif state == "AWAIT_RESET_WARNING":
                    if len(tokens) < 2:
                        raise ValueError("Нужно передать chat_id и user_id")
                    target_chat_id = self._parse_int(tokens[0], "chat_id")
                    user_id = self._parse_int(tokens[1], "user_id")
                    ok = self.store.reset_warnings(target_chat_id, user_id)
                    msg = "сброшен" if ok else "не найден"
                    self._send_to_chat(chat_id, f"Счетчик предупреждений пользователя {user_id} {msg}.", reply_markup=self._reply_keyboard("root"))
            except ValueError as exc:
                self._send_to_chat(chat_id, f"Ошибка: {exc}", reply_markup=self._reply_keyboard("root"))
            except Exception:
                logger.exception("Failed to process admin session step '%s'", state)
                self._send_to_chat(chat_id, "Внутренняя ошибка.", reply_markup=self._reply_keyboard("root"))
            finally:
                self._clear_session(chat_id)
            return

        # 3) Fallback to slash-commands for power users
        if not command.startswith("/"):
            # not a button, not a session, not a command — show menu
            self._show_root_menu(chat_id)
            return
        handlers = {
            "/start": self._cmd_help,
            "/help": self._cmd_help,
            "/add_chat": self._cmd_add_chat,
            "/remove_chat": self._cmd_remove_chat,
            "/list_chats": self._cmd_list_chats,
            "/add_keyword": self._cmd_add_keyword,
            "/remove_keyword": self._cmd_remove_keyword,
            "/list_keywords": self._cmd_list_keywords,
            "/warnings": self._cmd_warnings,
            "/reset_warning": self._cmd_reset_warning,
        }
        handler = handlers.get(command)
        if not handler:
            self._send_to_chat(chat_id, "Неизвестная команда. Нажмите 'Помощь' или 'Меню'.", reply_markup=self._main_keyboard())
            return
        try:
            handler(chat_id, args)
        except ValueError as exc:
            self._send_to_chat(chat_id, f"Ошибка: {exc}", reply_markup=self._main_keyboard())
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to process admin command '%s'", command)
            self._send_to_chat(chat_id, "Произошла внутренняя ошибка при обработке команды.", reply_markup=self._main_keyboard())

    def _handle_moderation(self, message: Dict[str, Any]) -> None:
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        chat_type = chat.get("type")
        if chat_id in self.config.admin_chat_ids:
            return
        if not self.store.is_chat_moderated(chat_id):
            return
        # Only in group contexts
        if chat_type not in ("group", "supergroup"):
            return

        # Anti-flood: enforce per-user message rate in a sliding window
        user = message.get("from", {})
        user_id = user.get("id")
        message_id = message.get("message_id")
        if user_id is None or message_id is None:
            return
        now = time.time()
        exceeded = False
        with self._rate_lock:
            dq = self._rate_buckets[chat_id][user_id]
            while dq and now - dq[0] > self.RL_WINDOW_SECONDS:
                dq.popleft()
            # Do not append a new timestamp if already at/over the limit within the window.
            # This prevents extending the cooldown indefinitely for active users.
            if len(dq) >= self.RL_MAX_MESSAGES and (dq and now - dq[0] <= self.RL_WINDOW_SECONDS):
                exceeded = True
            else:
                dq.append(now)
        if exceeded:
            try:
                self.api.delete_message(chat_id, message_id)
            except TelegramAPIError as exc:
                logger.warning("Failed to delete flood message %s in chat %s: %s", message_id, chat_id, exc)
                return
            self._send_ephemeral(chat_id, "Не флуди!")
            return

        text = self._extract_text(message)
        if not text:
            return

        # Detect overly long messages
        is_long = len(text) > self.MAX_MESSAGE_LENGTH
        if is_long:
            # For long messages: just delete without warnings/ban and notify with mention
            try:
                self.api.delete_message(chat_id, message.get("message_id"))
            except TelegramAPIError as exc:
                logger.warning("Failed to delete long message %s in chat %s: %s", message.get("message_id"), chat_id, exc)
                return
            mention_text, parse_mode = self._build_mention(message.get("from", {}))
            self._send_ephemeral(chat_id, f"{mention_text}, сообщение слишком длинное. Сократите, пожалуйста.", parse_mode=parse_mode)
            return

        # Detect forbidden keywords (when configured)
        matched: List[str] = []
        keywords = self.store.get_keywords(chat_id)
        if keywords:
            text_cf = text.casefold()
            matched = [kw for kw in keywords if kw.casefold() in text_cf]

        if matched:
            self._process_violation(message, matched)

    def _process_violation(self, message: Dict[str, Any], matched_keywords: List[str]) -> None:
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        message_id = message.get("message_id")
        user = message.get("from", {})
        user_id = user.get("id")
        if user_id is None or chat_id is None or message_id is None:
            return

        user_display = self._format_user(user)
        mention_text, parse_mode = self._build_mention(user)

        try:
            self.api.delete_message(chat_id, message_id)
        except TelegramAPIError as exc:
            logger.warning("Failed to delete message %s in chat %s: %s", message_id, chat_id, exc)
            return

        warning_count = self.store.increment_warning(chat_id, user_id)
        limit = self.config.warning_limit
        detected = matched_keywords[0] if matched_keywords else "запрещенное слово"
        detected_safe = self._escape_html(detected)
        warning_text = (
            f"{mention_text}, вам вынесено предупреждение за нарушение: «{detected_safe}». "
            f"Пожалуйста, соблюдайте правила чата. Предупреждение {warning_count} из {limit}."
        )

        self._send_ephemeral(chat_id, warning_text, parse_mode=parse_mode)

        if warning_count >= limit:
            self._enforce_ban(chat_id, user_id, user_display, warning_count)

    def _enforce_ban(self, chat_id: int, user_id: int, user_display: str, warning_count: int) -> None:
        # Avoid trying to ban admins/owner first
        if not self._can_ban(chat_id, user_id):
            info = f"Не удалось заблокировать {user_display}: админ или недостаточно прав."
            self._notify_admins(info)
            try:
                self.api.send_message(chat_id, f"Не могу заблокировать {user_display}: недостаточно прав.")
            except TelegramAPIError:
                pass
            return
        try:
            self.api.ban_chat_member(chat_id, user_id)
        except TelegramAPIError as exc:
            logger.warning("Failed to ban user %s in chat %s: %s", user_id, chat_id, exc)
            self._notify_admins(f"Ошибка блокировки {user_display} (id={user_id}) в чате {chat_id}: {exc}")
            return

        self.store.reset_warnings(chat_id, user_id)

        ban_text = f"{user_display} заблокирован после {warning_count} предупреждений."
        self._send_ephemeral(chat_id, ban_text)

        admin_note = f"Пользователь {user_display} (id={user_id}) заблокирован в чате {chat_id}."
        self._notify_admins(admin_note)

    def _cmd_help(self, chat_id: int, _: List[str]) -> None:
        commands = [
            "/add_chat <chat_id> [описание]",
            "/remove_chat <chat_id>",
            "/list_chats",
            "/add_keyword <слова через запятую>",
            "/remove_keyword <слова через запятую>",
            "/list_keywords",
            "/warnings <chat_id> [user_id]",
            "/reset_warning <chat_id> <user_id>",
        ]
        text = "Выберите раздел: Чаты или Слова. Доступные команды:\n" + "\n".join(commands)
        self._show_root_menu(chat_id, text=text)

    def _cmd_add_chat(self, chat_id: int, args: List[str]) -> None:
        if not args:
            raise ValueError("Укажите идентификатор чата")
        target_chat_id = self._parse_int(args[0], "chat_id")
        title = " ".join(args[1:]) if len(args) > 1 else None
        created = self.store.add_chat(target_chat_id, title)
        if created:
            self._send_to_chat(chat_id, f"Чат {target_chat_id} добавлен в список модерируемых.", reply_markup=self._reply_keyboard())
        else:
            self._send_to_chat(chat_id, f"Чат {target_chat_id} уже был в списке. Данные обновлены.", reply_markup=self._reply_keyboard())

    def _cmd_remove_chat(self, chat_id: int, args: List[str]) -> None:
        if not args:
            raise ValueError("Укажите идентификатор чата")
        target_chat_id = self._parse_int(args[0], "chat_id")
        if self.store.remove_chat(target_chat_id):
            self._send_to_chat(chat_id, f"Чат {target_chat_id} удален из списка модерируемых.", reply_markup=self._reply_keyboard())
        else:
            self._send_to_chat(chat_id, f"Чат {target_chat_id} не найден в списке.", reply_markup=self._reply_keyboard())

    def _cmd_list_chats(self, chat_id: int, _: List[str]) -> None:
        chats = self.store.list_chats()
        if not chats:
            self._send_to_chat(chat_id, "Нет модерируемых чатов.", reply_markup=self._reply_keyboard())
            return
        lines = []
        for target_chat_id, payload in chats.items():
            title, username = self._resolve_chat_title_username(target_chat_id, payload.get("title") or "")
            display = title or (f"@{username}" if username else "Без названия")
            lines.append(display)
        self._send_to_chat(chat_id, "Модерируемые чаты:\n" + "\n".join(lines), reply_markup=self._reply_keyboard("chats"))

    def _cmd_add_keyword(self, chat_id: int, args: List[str]) -> None:
        if not args:
            raise ValueError("Укажите слова через запятую")
        raw = " ".join(args)
        words = self._parse_words_csv(raw)
        added = self.store.add_keywords(words)
        if added:
            self._send_to_chat(chat_id, f"Добавлено слов: {added}.", reply_markup=self._reply_keyboard("words"))
        else:
            self._send_to_chat(chat_id, "Новые слова не обнаружены.", reply_markup=self._reply_keyboard("words"))

    def _cmd_remove_keyword(self, chat_id: int, args: List[str]) -> None:
        if not args:
            raise ValueError("Укажите слова через запятую для удаления")
        raw = " ".join(args)
        words = self._parse_words_csv(raw)
        removed = self.store.remove_keywords(words)
        if removed:
            self._send_to_chat(chat_id, f"Удалено слов: {removed}.", reply_markup=self._reply_keyboard("words"))
        else:
            self._send_to_chat(chat_id, "Совпадений для удаления не найдено.", reply_markup=self._reply_keyboard("words"))

    def _cmd_list_keywords(self, chat_id: int, args: List[str]) -> None:
        keywords = self.store.list_global_keywords()
        if not keywords:
            self._send_to_chat(chat_id, "Список ключевых слов пуст.", reply_markup=self._reply_keyboard("words"))
            return
        self._send_to_chat(chat_id, "Ключевые слова:\n" + "\n".join(keywords), reply_markup=self._reply_keyboard("words"))

    def _cmd_warnings(self, chat_id: int, args: List[str]) -> None:
        if not args:
            raise ValueError("Укажите chat_id")
        target_chat_id = self._parse_int(args[0], "chat_id")
        if len(args) == 1:
            warnings = self.store.get_all_warnings(target_chat_id)
            if not warnings:
                self._send_to_chat(chat_id, f"Для чата {target_chat_id} нет предупреждений.", reply_markup=self._reply_keyboard())
                return
            lines = [f"{user_id}: {count}" for user_id, count in warnings.items()]
            self._send_to_chat(chat_id, f"Предупреждения в чате {target_chat_id}:\n" + "\n".join(lines), reply_markup=self._reply_keyboard())
        else:
            user_id = self._parse_int(args[1], "user_id")
            count = self.store.get_warning(target_chat_id, user_id)
            self._send_to_chat(chat_id, f"Пользователь {user_id} имеет {count} предупреждений в чате {target_chat_id}.", reply_markup=self._reply_keyboard())

    def _cmd_reset_warning(self, chat_id: int, args: List[str]) -> None:
        if len(args) < 2:
            raise ValueError("Укажите chat_id и user_id")
        target_chat_id = self._parse_int(args[0], "chat_id")
        user_id = self._parse_int(args[1], "user_id")
        if self.store.reset_warnings(target_chat_id, user_id):
            self._send_to_chat(chat_id, f"Счетчик предупреждений пользователя {user_id} сброшен.", reply_markup=self._reply_keyboard())
        else:
            self._send_to_chat(chat_id, "Предупреждений не найдено или чат не модерируется.", reply_markup=self._reply_keyboard())

    def _notify_admins(self, text: str) -> None:
        for admin_chat in self.config.admin_chat_ids:
            try:
                self.api.send_message(admin_chat, text)
            except TelegramAPIError as exc:
                logger.warning("Failed to notify admin chat %s: %s", admin_chat, exc)
    
    def _can_ban(self, chat_id: int, user_id: int) -> bool:
        """Return True if it's reasonable to attempt a ban (not an admin/owner).
        If fetching admins fails, returns True to let API decide.
        """
        try:
            admins = self.api.get_chat_administrators(chat_id) or []
        except TelegramAPIError:
            return True
        for adm in admins:
            user = adm.get("user", {})
            if int(user.get("id", 0)) == int(user_id):
                return False
        return True

    def _resolve_chat_title_username(self, chat_id: int, cached_title: str = "") -> (str, Optional[str]):
        """Fetch chat title and @username when available; fall back to cached title."""
        try:
            info = self.api.get_chat(chat_id)
            title = info.get("title") or cached_title or ""
            username = info.get("username")
            if title and title != cached_title:
                try:
                    self.store.update_chat_title(chat_id, title)
                except Exception:
                    pass
            return title, username
        except TelegramAPIError:
            return cached_title, None

    def _extract_text(self, message: Dict[str, Any]) -> Optional[str]:
        if message.get("text"):
            return message["text"]
        if message.get("caption"):
            return message["caption"]
        return None

    def _format_user(self, user: Dict[str, Any]) -> str:
        username = user.get("username")
        if username:
            return f"@{username}"
        first_name = user.get("first_name")
        last_name = user.get("last_name")
        if first_name and last_name:
            return f"{first_name} {last_name}"
        if first_name:
            return first_name
        return str(user.get("id", "Пользователь"))

    def _build_mention(self, user: Dict[str, Any]) -> (str, Optional[str]):
        """Return mention text and parse mode.

        - If the user has a username, use @username (no parse mode).
        - Otherwise, use an HTML link tg://user?id=... so it pings the user.
        """
        user_id = user.get("id")
        username = user.get("username")
        display = self._format_user(user)
        if username:
            return f"@{username}", None
        safe = self._escape_html(display)
        return f"<a href=\"tg://user?id={user_id}\">{safe}</a>", "HTML"

    def _escape_html(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _parse_words_csv(self, raw: str) -> List[str]:
        """Parse a comma-separated list of words, trim whitespace and deduplicate case-insensitively.

        Examples:
        - "слово1, слово2,слово1" -> ["слово1", "слово2"]
        - Empty items are ignored.
        """
        parts = [p.strip() for p in (raw or "").split(",")]
        parts = [p for p in parts if p]
        seen = set()
        result: List[str] = []
        for p in parts:
            key = p.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(p)
        return result

    def _parse_int(self, raw: str, field: str) -> int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Некорректное число для {field}: {raw}") from exc

    def _send_to_chat(self, chat_id: int, text: str, reply_markup: Optional[Dict[str, Any]] = None, parse_mode: Optional[str] = None) -> None:
        try:
            self.api.send_message(chat_id, text, reply_markup=reply_markup, parse_mode=parse_mode)
        except TelegramAPIError as exc:
            logger.warning("Failed to send message to chat %s: %s", chat_id, exc)

    def _send_ephemeral(self, chat_id: int, text: str, reply_markup: Optional[Dict[str, Any]] = None, parse_mode: Optional[str] = None, ttl_seconds: int = 20) -> None:
        """Send a message and schedule its deletion after ttl_seconds.

        Используется для сообщений модерации в группах (предупреждения, ответы на флуд).
        Админ-уведомления и UI через этот метод не отправляются.
        """
        try:
            result = self.api.send_message(chat_id, text, reply_markup=reply_markup, parse_mode=parse_mode)
        except TelegramAPIError as exc:
            logger.warning("Failed to send ephemeral message to chat %s: %s", chat_id, exc)
            return
        try:
            message_id = int(result.get("message_id")) if isinstance(result, dict) else None
        except Exception:
            message_id = None
        if message_id is None:
            return
        def _delete_later() -> None:
            try:
                self.api.delete_message(chat_id, message_id)
            except TelegramAPIError:
                pass
            except Exception:
                logger.debug("Silent ignore: failed to delete ephemeral message %s in chat %s", message_id, chat_id)
        timer = threading.Timer(ttl_seconds, _delete_later)
        timer.daemon = True
        timer.start()

    # --- Admin session helpers and keyboard layout ---
    def _set_session(self, chat_id: int, state: str, data: Optional[Dict[str, Any]] = None) -> None:
        with self._admin_sessions_lock:
            self._admin_sessions[chat_id] = {"state": state, "data": data or {}}

    def _get_session(self, chat_id: int) -> Optional[str]:
        with self._admin_sessions_lock:
            sess = self._admin_sessions.get(chat_id)
            return sess.get("state") if sess else None

    def _get_session_obj(self, chat_id: int) -> Optional[Dict[str, Any]]:
        with self._admin_sessions_lock:
            return self._admin_sessions.get(chat_id)

    def _clear_session(self, chat_id: int) -> None:
        with self._admin_sessions_lock:
            self._admin_sessions.pop(chat_id, None)

    def _reply_keyboard(self, mode: str = "root") -> Dict[str, Any]:
        if mode == "chats":
            keyboard = [
                [{"text": self.BTN_LIST_CHATS}],
                [
                    {
                        "text": self.BTN_ADD_CHAT,
                        "request_chat": {
                            "request_id": self.REQ_ADD_CHAT,
                            "chat_is_channel": False,
                            "bot_is_member": True,
                        },
                    },
                    {
                        "text": self.BTN_REMOVE_CHAT,
                        "request_chat": {
                            "request_id": self.REQ_REMOVE_CHAT,
                            "chat_is_channel": False,
                            "bot_is_member": True,
                        },
                    },
                ],
                [{"text": self.BTN_BACK}],
            ]
        elif mode == "words":
            keyboard = [
                [
                    {"text": self.BTN_ADD_WORD},
                    {"text": self.BTN_REMOVE_WORD},
                    {"text": self.BTN_LIST_WORDS},
                ],
                [{"text": self.BTN_BACK}],
            ]
        else:  # root
            keyboard = [[
                {"text": self.BTN_GROUP_CHATS},
                {"text": self.BTN_GROUP_WORDS},
            ]]
        return {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": False}

    # Inline keyboard builders
    def _inline_root(self) -> Dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": self.BTN_GROUP_CHATS, "callback_data": "menu:chats"},
                    {"text": self.BTN_GROUP_WORDS, "callback_data": "menu:words"},
                ],
                [
                    {"text": self.BTN_HELP, "callback_data": "menu:help"},
                ],
            ]
        }

    def _inline_chats(self) -> Dict[str, Any]:
        return {
            "inline_keyboard": [
                [{"text": self.BTN_LIST_CHATS, "callback_data": "chats:list"}],
                [
                    {"text": self.BTN_ADD_CHAT, "callback_data": "chats:add"},
                    {"text": self.BTN_REMOVE_CHAT, "callback_data": "chats:remove"},
                ],
                [{"text": "⬅ Назад", "callback_data": "menu:root"}],
            ]
        }

    def _inline_words(self) -> Dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": self.BTN_ADD_WORD, "callback_data": "words:add"},
                    {"text": self.BTN_REMOVE_WORD, "callback_data": "words:remove"},
                ],
                [{"text": self.BTN_LIST_WORDS, "callback_data": "words:list"}],
                [{"text": "⬅ Назад", "callback_data": "menu:root"}],
            ]
        }

    def _show_root_menu(self, chat_id: int, text: str = "Главное меню:") -> None:
        # Show reply keyboard with grouped buttons
        self._send_to_chat(chat_id, text, reply_markup=self._reply_keyboard("root"))

    def _show_chats_menu(self, chat_id: int, text: str = "Управление чатами:") -> None:
        self._send_to_chat(chat_id, text, reply_markup=self._reply_keyboard("chats"))

    def _show_words_menu(self, chat_id: int, text: str = "Управление словами:") -> None:
        self._send_to_chat(chat_id, text, reply_markup=self._reply_keyboard("words"))

    def _show_global_keywords(self, chat_id: int) -> None:
        keywords = self.store.list_global_keywords()
        if not keywords:
            self._send_to_chat(chat_id, "Список ключевых слов пуст.", reply_markup=self._reply_keyboard("words"))
            return
        self._send_to_chat(chat_id, "Ключевые слова:\n" + "\n".join(keywords), reply_markup=self._reply_keyboard("words"))

    def _upsert_menu(self, chat_id: int, text: str, reply_markup: Dict[str, Any]) -> None:
        with self._admin_ui_lock:
            msg_id = self._admin_menu_message.get(chat_id)
        if msg_id:
            try:
                self.api.edit_message_text(chat_id, msg_id, text, reply_markup=reply_markup)
                return
            except TelegramAPIError:
                # fallback to sending a new menu
                pass
        try:
            result = self.api.send_message(chat_id, text, reply_markup=reply_markup)
            message_id = int(result.get("message_id")) if isinstance(result, dict) else None
            if message_id:
                with self._admin_ui_lock:
                    self._admin_menu_message[chat_id] = message_id
        except TelegramAPIError as exc:
            logger.warning("Failed to send inline menu to chat %s: %s", chat_id, exc)

    def _handle_callback(self, callback: Dict[str, Any]) -> None:
        data = callback.get("data") or ""
        message = callback.get("message") or {}
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        callback_id = callback.get("id")
        # Always answer to stop the loader
        try:
            self.api.answer_callback_query(callback_id)
        except TelegramAPIError:
            pass

        if not chat_id:
            return
        if data in ("menu:root", "menu:start"):
            self._clear_session(chat_id)
            self._show_root_menu(chat_id)
            return
        if data == "menu:chats":
            self._clear_session(chat_id)
            self._show_chats_menu(chat_id)
            return
        if data == "menu:words":
            self._clear_session(chat_id)
            self._show_words_menu(chat_id)
            return
        if data == "menu:warnings":
            self._clear_session(chat_id)
            self._upsert_menu(chat_id, "Введите: Предупреждения → отправьте chat_id [user_id]", self._inline_root())
            self._set_session(chat_id, "AWAIT_WARNINGS")
            return
        if data == "menu:reset":
            self._clear_session(chat_id)
            self._upsert_menu(chat_id, "Введите: Сброс предупреждений → chat_id user_id", self._inline_root())
            self._set_session(chat_id, "AWAIT_RESET_WARNING")
            return
        if data == "menu:help":
            self._cmd_help(chat_id, [])
            return

        if data == "chats:list":
            chats = self.store.list_chats()
            if not chats:
                self._upsert_menu(chat_id, "Нет модерируемых чатов.", self._inline_chats())
                return
            lines = []
            for target_chat_id, payload in chats.items():
                title = payload.get("title") or "(без описания)"
                keywords = payload.get("keywords") or []
                lines.append(
                    f"{target_chat_id} — {title}, ключевых слов: {len(keywords)}, предупреждений: {len(payload.get('warnings', {}))}"
                )
            self._upsert_menu(chat_id, "Модерируемые чаты:\n" + "\n".join(lines), self._inline_chats())
            return
        if data in ("chats:add", "chats:remove"):
            # Inform that bottom keyboard already has request buttons
            self._send_to_chat(chat_id, "Нажмите кнопку внизу: Добавить чат/Удалить чат.", reply_markup=self._reply_keyboard())
            return

        if data.startswith("words:"):
            if data == "words:add":
                self._set_session(chat_id, "AWAIT_ADD_KEYWORD")
                self._send_to_chat(chat_id, "Отправьте слова через запятую", reply_markup=self._reply_keyboard("words"))
            elif data == "words:remove":
                self._set_session(chat_id, "AWAIT_REMOVE_KEYWORD")
                self._send_to_chat(chat_id, "Отправьте слова через запятую для удаления", reply_markup=self._reply_keyboard("words"))
            elif data == "words:list":
                self._show_global_keywords(chat_id)
            return

    def _maybe_delete_ui_echo(self, chat_id: int, message_id: Optional[int], text: str) -> None:
        if message_id is None:
            return
        if text in {
            self.BTN_GROUP_CHATS,
            self.BTN_LIST_CHATS,
            self.BTN_ADD_CHAT,
            self.BTN_REMOVE_CHAT,
            self.BTN_GROUP_WORDS,
            self.BTN_ADD_WORD,
            self.BTN_REMOVE_WORD,
            self.BTN_LIST_WORDS,
            self.BTN_WARNINGS,
            self.BTN_RESET_WARNING,
            self.BTN_HELP,
            self.BTN_MENU,
        }:
            try:
                self.api.delete_message(chat_id, message_id)
            except TelegramAPIError:
                pass


__all__ = ["ModerationBot"]
