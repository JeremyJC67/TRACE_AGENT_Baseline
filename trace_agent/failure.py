"""Failure taxonomy and classification helpers."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from browser_use.browser.views import BrowserStateHistory


class FailureType(str, Enum):
	"""Standardized failure categories for trace analysis."""

	TIMEOUT = 'Timeout'
	ELEMENT_NOT_FOUND = 'ElementNotFound'
	CLICK_FAILED = 'ClickFailed'
	CAPTCHA = 'Captcha'
	RATE_LIMIT = 'RateLimit'
	NAVIGATION_ERROR = 'NavigationError'
	AUTH_REQUIRED = 'AuthRequired'
	OTHER = 'Other'


def _contains_any(value: str, needles: Iterable[str]) -> bool:
	return any(needle in value for needle in needles)


def classify_failure(error_msg: str | None, state: BrowserStateHistory | None = None) -> FailureType | None:
	"""Classify a failure based on error message and optional browser state."""
	if not error_msg:
		return None

	message = error_msg.lower()

	if _contains_any(message, ['timeout', 'timed out', 'time out', 'page readiness timeout']):
		return FailureType.TIMEOUT
	if _contains_any(message, ['captcha', 'unusual traffic', 'robot check', 'are you a robot']):
		return FailureType.CAPTCHA
	if _contains_any(message, ['rate limit', 'too many requests', '429']):
		return FailureType.RATE_LIMIT
	if _contains_any(message, ['navigation', 'net::', 'err_']):
		return FailureType.NAVIGATION_ERROR
	if _contains_any(message, ['element', 'no node', 'not found', 'not available']):
		return FailureType.ELEMENT_NOT_FOUND
	if _contains_any(message, ['click', 'clicked', 'failed to click']):
		return FailureType.CLICK_FAILED

	if state:
		url = state.url.lower() if state.url else ''
		title = state.title.lower() if state.title else ''
		if _contains_any(url, ['login', 'signin', 'sign-in', 'auth']) or _contains_any(
			title, ['login', 'sign in', 'authentication']
		):
			return FailureType.AUTH_REQUIRED

	return FailureType.OTHER
