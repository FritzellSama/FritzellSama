"""
Telegram Notifier - Telegram bot notification channel.

Sends notifications via Telegram bot API with retry logic,
rate limiting, and message formatting.
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from decimal import Decimal
import os
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class TelegramMessage:
    """Telegram message structure."""
    chat_id: str
    text: str
    parse_mode: str = "Markdown"
    disable_web_page_preview: bool = True
    disable_notification: bool = False


class TelegramNotifier:
    """
    Telegram notification channel implementation.

    Sends notifications to Telegram using the Bot API with proper
    error handling, retry logic, and rate limiting.

    Attributes:
        config: Configuration dictionary from environment
        bot_token: Telegram bot token
        default_chat_id: Default chat ID for notifications
        session: aiohttp session for API calls
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize Telegram notifier.

        Args:
            config: Optional configuration override. If None, loads from environment.

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config or self._load_config()
        self._validate_config()

        self.bot_token = self.config["bot_token"]
        self.default_chat_id = self.config.get("default_chat_id")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Retry settings
        self._max_retries = int(self.config.get(
            "max_retries",
            os.getenv("TELEGRAM_MAX_RETRIES", "3")
        ))
        self._retry_delay_base = float(self.config.get(
            "retry_delay_base",
            os.getenv("TELEGRAM_RETRY_DELAY_BASE", "2")
        ))
        self._retry_delay_max = float(self.config.get(
            "retry_delay_max",
            os.getenv("TELEGRAM_RETRY_DELAY_MAX", "60")
        ))

        # Timeout settings
        self._timeout_seconds = float(self.config.get(
            "timeout_seconds",
            os.getenv("TELEGRAM_TIMEOUT_SECONDS", "30")
        ))

        # Message settings
        self._max_message_length = int(self.config.get(
            "max_message_length",
            os.getenv("TELEGRAM_MAX_MESSAGE_LENGTH", "4096")
        ))

        logger.info(
            "telegram_notifier_initialized",
            default_chat_id=self.default_chat_id,
            max_retries=self._max_retries
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "default_chat_id": os.getenv("TELEGRAM_DEFAULT_CHAT_ID"),
            "max_retries": os.getenv("TELEGRAM_MAX_RETRIES", "3"),
            "retry_delay_base": os.getenv("TELEGRAM_RETRY_DELAY_BASE", "2"),
            "retry_delay_max": os.getenv("TELEGRAM_RETRY_DELAY_MAX", "60"),
            "timeout_seconds": os.getenv("TELEGRAM_TIMEOUT_SECONDS", "30"),
            "max_message_length": os.getenv("TELEGRAM_MAX_MESSAGE_LENGTH", "4096"),
            "parse_mode": os.getenv("TELEGRAM_PARSE_MODE", "Markdown"),
            "disable_web_preview": os.getenv("TELEGRAM_DISABLE_WEB_PREVIEW", "true").lower() == "true",
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.config.get("bot_token"):
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        if int(self.config.get("max_retries", 3)) < 0:
            raise ValueError("max_retries must be >= 0")

        if float(self.config.get("timeout_seconds", 30)) <= 0:
            raise ValueError("timeout_seconds must be > 0")

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.

        Returns:
            Active aiohttp session
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        return self._session

    async def send(self, notification: Any) -> None:
        """
        Send notification via Telegram.

        Args:
            notification: Notification object with message and metadata

        Raises:
            Exception: If all retry attempts fail
        """
        try:
            # Extract message and metadata
            message_text = notification.message if hasattr(notification, 'message') else str(notification)
            metadata = notification.metadata if hasattr(notification, 'metadata') else {}

            chat_id = metadata.get("chat_id") or self.default_chat_id

            if not chat_id:
                raise ValueError("No chat_id specified and no default_chat_id configured")

            # Create Telegram message
            telegram_msg = TelegramMessage(
                chat_id=str(chat_id),
                text=message_text,
                parse_mode=metadata.get("parse_mode", self.config.get("parse_mode", "Markdown")),
                disable_web_page_preview=metadata.get(
                    "disable_web_preview",
                    self.config.get("disable_web_preview", True)
                ),
                disable_notification=metadata.get("silent", False)
            )

            # Send message with retry logic
            await self._send_message_with_retry(telegram_msg)

            logger.info(
                "telegram_notification_sent",
                chat_id=chat_id,
                message_length=len(message_text)
            )

        except Exception as e:
            logger.error("telegram_notification_failed", error=str(e))
            raise

    async def _send_message_with_retry(self, message: TelegramMessage) -> Dict[str, Any]:
        """
        Send Telegram message with exponential backoff retry.

        Args:
            message: Telegram message to send

        Returns:
            API response dictionary

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._send_message(message)
                return response

            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(
                    "telegram_send_failed_retrying",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e)
                )

                if attempt < self._max_retries:
                    # Exponential backoff
                    delay = min(
                        self._retry_delay_base ** (attempt + 1),
                        self._retry_delay_max
                    )
                    await asyncio.sleep(delay)
                else:
                    # Max retries reached
                    logger.error(
                        "telegram_send_failed_max_retries",
                        max_retries=self._max_retries,
                        error=str(e)
                    )
                    raise

            except Exception as e:
                # Non-retryable error
                logger.error("telegram_send_error", error=str(e))
                raise

        # Should not reach here, but raise last exception if we do
        if last_exception:
            raise last_exception

    async def _send_message(self, message: TelegramMessage) -> Dict[str, Any]:
        """
        Send message to Telegram API.

        Args:
            message: Telegram message to send

        Returns:
            API response dictionary

        Raises:
            aiohttp.ClientError: If HTTP request fails
            ValueError: If API returns error
        """
        session = await self._get_session()

        # Split message if too long
        messages = self._split_message(message.text)

        last_response = None

        for msg_part in messages:
            payload = {
                "chat_id": message.chat_id,
                "text": msg_part,
                "parse_mode": message.parse_mode,
                "disable_web_page_preview": message.disable_web_page_preview,
                "disable_notification": message.disable_notification,
            }

            url = f"{self.api_url}/sendMessage"

            try:
                async with session.post(url, json=payload) as response:
                    response_data = await response.json()

                    if not response_data.get("ok"):
                        error_description = response_data.get("description", "Unknown error")
                        logger.error(
                            "telegram_api_error",
                            error=error_description,
                            error_code=response_data.get("error_code")
                        )
                        raise ValueError(f"Telegram API error: {error_description}")

                    last_response = response_data

                    logger.debug(
                        "telegram_message_sent",
                        chat_id=message.chat_id,
                        message_id=response_data.get("result", {}).get("message_id")
                    )

                    # Small delay between messages to avoid rate limiting
                    if len(messages) > 1:
                        await asyncio.sleep(0.5)

            except aiohttp.ClientError as e:
                logger.error("telegram_http_error", error=str(e))
                raise

        return last_response or {}

    def _split_message(self, text: str) -> List[str]:
        """
        Split long message into multiple parts.

        Args:
            text: Message text to split

        Returns:
            List of message parts
        """
        if len(text) <= self._max_message_length:
            return [text]

        parts = []
        current_part = ""

        for line in text.split('\n'):
            if len(current_part) + len(line) + 1 <= self._max_message_length:
                current_part += line + '\n'
            else:
                if current_part:
                    parts.append(current_part.strip())
                current_part = line + '\n'

        if current_part:
            parts.append(current_part.strip())

        # If a single line is too long, force split
        final_parts = []
        for part in parts:
            if len(part) <= self._max_message_length:
                final_parts.append(part)
            else:
                # Force split at max length
                for i in range(0, len(part), self._max_message_length):
                    final_parts.append(part[i:i + self._max_message_length])

        return final_parts

    async def send_text(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
        silent: bool = False
    ) -> None:
        """
        Send plain text message.

        Args:
            text: Message text
            chat_id: Optional chat ID override
            parse_mode: Optional parse mode override
            silent: Send silently without notification

        Raises:
            Exception: If send fails
        """
        target_chat_id = chat_id or self.default_chat_id

        if not target_chat_id:
            raise ValueError("No chat_id specified and no default_chat_id configured")

        message = TelegramMessage(
            chat_id=str(target_chat_id),
            text=text,
            parse_mode=parse_mode or self.config.get("parse_mode", "Markdown"),
            disable_notification=silent
        )

        await self._send_message_with_retry(message)

    async def send_to_multiple(
        self,
        text: str,
        chat_ids: List[str],
        parse_mode: Optional[str] = None,
        silent: bool = False
    ) -> None:
        """
        Send message to multiple chats.

        Args:
            text: Message text
            chat_ids: List of chat IDs
            parse_mode: Optional parse mode override
            silent: Send silently without notification
        """
        tasks = []

        for chat_id in chat_ids:
            message = TelegramMessage(
                chat_id=str(chat_id),
                text=text,
                parse_mode=parse_mode or self.config.get("parse_mode", "Markdown"),
                disable_notification=silent
            )
            tasks.append(self._send_message_with_retry(message))

        # Send all messages concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "telegram_broadcast_failed",
                    chat_id=chat_ids[i],
                    error=str(result)
                )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("telegram_notifier_session_closed")

    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            session = await self._get_session()
            url = f"{self.api_url}/getMe"

            async with session.get(url) as response:
                data = await response.json()

                if data.get("ok"):
                    bot_info = data.get("result", {})
                    logger.info(
                        "telegram_connection_test_success",
                        bot_username=bot_info.get("username"),
                        bot_id=bot_info.get("id")
                    )
                    return True
                else:
                    logger.error(
                        "telegram_connection_test_failed",
                        error=data.get("description")
                    )
                    return False

        except Exception as e:
            logger.error("telegram_connection_test_error", error=str(e))
            return False

    async def get_updates(self, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get bot updates from Telegram.

        Args:
            offset: Update offset for pagination

        Returns:
            List of updates

        Raises:
            Exception: If request fails
        """
        try:
            session = await self._get_session()
            url = f"{self.api_url}/getUpdates"

            params = {}
            if offset is not None:
                params["offset"] = offset

            async with session.get(url, params=params) as response:
                data = await response.json()

                if data.get("ok"):
                    return data.get("result", [])
                else:
                    raise ValueError(f"Telegram API error: {data.get('description')}")

        except Exception as e:
            logger.error("telegram_get_updates_error", error=str(e))
            raise
