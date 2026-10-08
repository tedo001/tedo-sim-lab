"""Keep credentials out of logs, exceptions and anything shown on screen.

Two layers:

* every credential value the :class:`~core.common.secrets.CredentialStore`
  hands out is registered here and replaced wherever it appears;
* well-known token shapes (Hugging Face, GitHub) and ``NAME=value``
  assignments whose name says key / token / secret / password are masked even
  when the value was never registered, e.g. a token pasted into a CSV path or
  echoed by a third-party library.

Long hex strings are deliberately *not* masked: commit SHAs and spec hashes
are hex and must stay readable.
"""

from __future__ import annotations

import logging
import re
import threading

__all__ = ["MASK", "SecretMaskingFilter", "mask_text", "register_secret", "forget_secrets"]

MASK = "********"
_MIN_SECRET_LENGTH = 4

_lock = threading.Lock()
_known: set[str] = set()

_TOKEN_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(?<=\bbearer )[A-Za-z0-9._~+/=-]{12,}"),
)

# NAME=value or NAME: value where NAME ends in a credential word. ``max_tokens``
# and ``tokenizer`` do not match: the credential word must end the name.
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|APIKEY|ACCESS_KEY|KAGGLE_KEY|TOKEN|SECRET|PASSWORD|PASSWD))"
    r"(\s*[=:]\s*)(['\"]?)([^\s'\",;]+)")


def register_secret(value: str | None) -> None:
    """Mask ``value`` from now on. Values shorter than 4 characters are ignored."""
    if value and len(value) >= _MIN_SECRET_LENGTH:
        with _lock:
            _known.add(value)


def forget_secrets() -> None:
    """Drop every registered value (tests only)."""
    with _lock:
        _known.clear()


def mask_text(text: str) -> str:
    """Return ``text`` with every known secret and token-shaped string masked."""
    if not text:
        return text
    with _lock:
        known = sorted(_known, key=len, reverse=True)
    for value in known:
        if value in text:
            text = text.replace(value, MASK)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(MASK, text)
    return _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{MASK}", text)


class SecretMaskingFilter(logging.Filter):
    """Masks the message, its arguments, the traceback and the stack of a record.

    Attach it to *handlers*: a filter on a logger does not see records that
    propagate up from child loggers.
    """

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a bad format string must not lose the record
            message = f"{record.msg!s} {record.args!r}"
        record.msg = mask_text(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = self._formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = mask_text(record.exc_text)
        if record.stack_info:
            record.stack_info = mask_text(record.stack_info)
        return True
