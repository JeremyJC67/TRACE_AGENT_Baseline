"""Best-effort HAR recorder using CDP Network events."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, Field


class HarHeader(BaseModel):
	"""HTTP header entry for HAR."""

	name: str
	value: str


class HarQueryString(BaseModel):
	"""Query string entry for HAR."""

	name: str
	value: str


class HarPostData(BaseModel):
	"""POST data payload for HAR."""

	mimeType: str
	text: str


class HarRequest(BaseModel):
	"""HAR request record."""

	method: str
	url: str
	httpVersion: str = 'HTTP/1.1'
	headers: list[HarHeader]
	queryString: list[HarQueryString] = Field(default_factory=list)
	cookies: list[dict[str, Any]] = Field(default_factory=list)
	headersSize: int = -1
	bodySize: int = -1
	postData: HarPostData | None = None


class HarContent(BaseModel):
	"""HAR response content payload."""

	size: int = 0
	mimeType: str = ''
	text: str | None = None
	encoding: str | None = None


class HarResponse(BaseModel):
	"""HAR response record."""

	status: int
	statusText: str
	httpVersion: str = 'HTTP/1.1'
	headers: list[HarHeader] = Field(default_factory=list)
	cookies: list[dict[str, Any]] = Field(default_factory=list)
	content: HarContent = Field(default_factory=HarContent)
	redirectURL: str = ''
	headersSize: int = -1
	bodySize: int = -1


class HarEntry(BaseModel):
	"""HAR entry record."""

	startedDateTime: str
	time: float = 0.0
	request: HarRequest
	response: HarResponse
	cache: dict[str, Any] = Field(default_factory=dict)
	timings: dict[str, Any] = Field(default_factory=lambda: {'send': 0, 'wait': 0, 'receive': 0})


class HarCreator(BaseModel):
	"""HAR creator metadata."""

	name: str = 'trace-agent'
	version: str = '0.1'


class HarLog(BaseModel):
	"""HAR log container."""

	version: str = '1.2'
	creator: HarCreator = Field(default_factory=HarCreator)
	entries: list[HarEntry] = Field(default_factory=list)


class HarFile(BaseModel):
	"""HAR file top-level schema."""

	log: HarLog


class HarRecorder:
	"""Capture network events and serialize as HAR."""

	def __init__(self, output_path: Path, fallback_url: str | None = None) -> None:
		self.output_path = output_path
		self.fallback_url = fallback_url
		self._entries: dict[str, HarEntry] = {}
		self._entry_order: list[str] = []
		self._request_start_ts: dict[str, float] = {}
		self._enabled = False
		self._cdp_session = None

	async def start(self, browser_session: Any) -> None:
		"""Enable CDP Network events and register listeners."""
		if self._enabled:
			return

		target_id = getattr(browser_session, 'agent_focus_target_id', None)
		cdp_session = await browser_session.get_or_create_cdp_session(target_id, focus=False)
		self._cdp_session = cdp_session

		await cdp_session.cdp_client.send.Network.enable(params={}, session_id=cdp_session.session_id)
		cdp_session.cdp_client.register.Network.requestWillBeSent(self._on_request_will_be_sent)
		cdp_session.cdp_client.register.Network.responseReceived(self._on_response_received)
		cdp_session.cdp_client.register.Network.loadingFailed(self._on_loading_failed)
		cdp_session.cdp_client.register.Network.loadingFinished(self._on_loading_finished)

		self._enabled = True

	def write(self) -> None:
		"""Write HAR data to disk, ensuring at least one entry."""
		entries = [self._entries[request_id] for request_id in self._entry_order if request_id in self._entries]
		if not entries:
			entries.append(self._synthetic_entry(self.fallback_url or 'about:blank'))

		har_file = HarFile(log=HarLog(entries=entries))
		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		payload = har_file.model_dump(mode='json')
		self.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')

	def _on_request_will_be_sent(self, event: dict[str, Any], session_id: str | None = None) -> None:
		request_id = event.get('requestId')
		if not request_id:
			return

		request = event.get('request') or {}
		url = request.get('url', '')
		method = request.get('method', 'GET')
		headers = _headers_to_list(request.get('headers'))
		query_params = _query_string_from_url(url)
		post_data = _post_data_from_request(request, headers)

		started = _epoch_to_iso(event.get('wallTime'))
		entry = HarEntry(
			startedDateTime=started,
			request=HarRequest(
				method=method,
				url=url,
				headers=headers,
				queryString=query_params,
				postData=post_data,
			),
			response=HarResponse(
				status=0,
				statusText='',
				headers=[],
				content=HarContent(size=0, mimeType=''),
			),
		)
		self._entries[request_id] = entry
		self._entry_order.append(request_id)
		if event.get('timestamp') is not None:
			self._request_start_ts[request_id] = float(event['timestamp'])

	def _on_response_received(self, event: dict[str, Any], session_id: str | None = None) -> None:
		request_id = event.get('requestId')
		if not request_id:
			return

		response = event.get('response') or {}
		entry = self._entries.get(request_id)
		if entry is None:
			entry = self._synthetic_entry(response.get('url', ''))
			self._entries[request_id] = entry
			self._entry_order.append(request_id)

		entry.response.status = int(response.get('status', 0))
		entry.response.statusText = str(response.get('statusText', '') or '')
		entry.response.headers = _headers_to_list(response.get('headers'))
		entry.response.redirectURL = _redirect_from_headers(response.get('headers'))
		entry.response.content = HarContent(
			size=int(response.get('encodedDataLength') or 0),
			mimeType=str(response.get('mimeType') or _header_value(entry.response.headers, 'content-type') or ''),
		)

	def _on_loading_failed(self, event: dict[str, Any], session_id: str | None = None) -> None:
		request_id = event.get('requestId')
		if not request_id or request_id not in self._entries:
			return

		entry = self._entries[request_id]
		entry.response.status = 0
		entry.response.statusText = str(event.get('errorText') or '')

	def _on_loading_finished(self, event: dict[str, Any], session_id: str | None = None) -> None:
		request_id = event.get('requestId')
		if not request_id or request_id not in self._entries:
			return

		if event.get('timestamp') is not None and request_id in self._request_start_ts:
			start_ts = self._request_start_ts[request_id]
			entry = self._entries[request_id]
			entry.time = max(0.0, (float(event['timestamp']) - start_ts) * 1000)

	def _synthetic_entry(self, url: str) -> HarEntry:
		headers = [HarHeader(name='accept', value='text/html')]
		started = _epoch_to_iso(None)
		return HarEntry(
			startedDateTime=started,
			request=HarRequest(method='GET', url=url, headers=headers),
			response=HarResponse(
				status=200,
				statusText='OK',
				headers=[HarHeader(name='content-type', value='text/html')],
				content=HarContent(size=0, mimeType='text/html'),
			),
		)


def _headers_to_list(headers: Any) -> list[HarHeader]:
	if isinstance(headers, list):
		entries = []
		for item in headers:
			name = item.get('name') if isinstance(item, dict) else None
			value = item.get('value') if isinstance(item, dict) else None
			if name is None or value is None:
				continue
			entries.append(HarHeader(name=str(name), value=str(value)))
		return entries
	if isinstance(headers, dict):
		return [HarHeader(name=str(name), value=str(value)) for name, value in headers.items()]
	return []


def _header_value(headers: list[HarHeader], name: str) -> str | None:
	target = name.lower()
	for header in headers:
		if header.name.lower() == target:
			return header.value
	return None


def _query_string_from_url(url: str) -> list[HarQueryString]:
	parsed = urlparse(url)
	return [HarQueryString(name=key, value=value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)]


def _post_data_from_request(request: dict[str, Any], headers: list[HarHeader]) -> HarPostData | None:
	post_text = request.get('postData')
	if not post_text:
		return None
	content_type = _header_value(headers, 'content-type') or 'application/octet-stream'
	return HarPostData(mimeType=str(content_type), text=str(post_text))


def _redirect_from_headers(headers: Any) -> str:
	if not isinstance(headers, dict):
		return ''
	for key, value in headers.items():
		if str(key).lower() == 'location':
			return str(value)
	return ''


def _epoch_to_iso(timestamp: float | None) -> str:
	if timestamp is None:
		timestamp = datetime.now(timezone.utc).timestamp()
	return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace('+00:00', 'Z')
