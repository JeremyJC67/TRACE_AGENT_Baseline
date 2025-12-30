"""Single-round runner for baseline agent with tracing."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from browser_use import Agent, Browser, ChatBrowserUse

from trace_agent.failure import FailureType, classify_failure
from trace_agent.har_recorder import HarRecorder
from trace_agent.trace_recorder import TraceRecorder
from trace_agent.integrations.webarena_verified_tasks import load_env_config, resolve_start_url


class TaskSpec(BaseModel):
	"""Minimal task spec for running a baseline agent."""

	task_id: str | None = None
	task: str
	start_url: str | None = None
	metadata: dict[str, Any] | None = None


class RunnerConfig(BaseModel):
	"""Runner configuration for output and agent settings."""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	output_root: Path = Path('runs')
	round_id: int = 0
	max_steps: int = 100
	headless: bool = True
	viewport_width: int = 1280
	viewport_height: int = 720
	enable_failure_classification: bool = False
	env_config_path: Path | None = None
	record_har: bool = False
	max_domain_resets: int = 1
	enable_webarena_verified_eval: bool = False
	webarena_verified_config_path: Path | None = None


class RunSummary(BaseModel):
	"""Summary of one run for analysis."""

	task_id: str
	round_id: int
	raw_start_url: str | None = None
	resolved_start_url: str | None = None
	allowed_domains: list[str] | None = None
	success: bool | None
	steps: int
	time_seconds: float | None
	final_url: str | None
	har_status: str | None = None
	har_missing_reason: str | None = None
	eval_status: str | None = None
	eval_reason: str | None = None
	failure_histogram: dict[str, int] = Field(default_factory=dict)


class AgentResponsePayload(BaseModel):
	"""Minimal agent response payload compatible with WebArena-Verified evaluators."""

	task_id: str
	task_type: str
	status: str
	retrieved_data: list[Any] | None = None
	error_details: str | None = None
	final_answer: str | None = None
	final_url: str | None = None
	steps: int
	action_summary: list[str]


_PLACEHOLDER_PATTERN = re.compile(r'__[^_]+__')
_SITE_TO_PLACEHOLDER = {
	'shopping_admin': '__SHOPPING_ADMIN__',
	'shopping': '__SHOPPING__',
	'gitlab': '__GITLAB__',
	'reddit': '__REDDIT__',
	'wikipedia': '__WIKIPEDIA__',
	'map': '__MAP__',
}


def run_task(task_spec: TaskSpec | dict[str, Any] | str, config: RunnerConfig | None = None) -> RunSummary:
	"""Run a baseline agent and write trace + summary to disk."""
	config = config or RunnerConfig()
	spec = _load_task_spec(task_spec)
	task_id = spec.task_id or f'task_{uuid4().hex[:8]}'

	output_dir = config.output_root / task_id / f'round_{config.round_id}'
	output_dir.mkdir(parents=True, exist_ok=True)

	raw_start_url = _extract_raw_start_url(spec)
	env_config = None
	if spec.start_url and _contains_placeholder(spec.start_url) and not config.env_config_path:
		raise ValueError('start_url contains placeholders but env_config_path was not provided.')
	if config.env_config_path:
		env_config = load_env_config(config.env_config_path)
		if spec.start_url and _contains_placeholder(spec.start_url):
			resolved_task = resolve_start_url(spec.model_dump(mode='json'), env_config)
			spec = TaskSpec.model_validate(resolved_task)

	resolved_start_url = spec.start_url
	allowed_domains = _derive_allowed_domains(resolved_start_url, env_config, spec.metadata)

	print(
		f'[runner] task_id={task_id} raw_start_url={raw_start_url} '
		f'resolved_start_url={resolved_start_url} allowed_domains={allowed_domains}'
	)

	failure_classifier = _build_failure_classifier(allowed_domains, config.enable_failure_classification)
	trace_recorder = TraceRecorder(
		output_dir=output_dir,
		failure_classifier=failure_classifier,
	)

	har_path = output_dir / 'network.har'
	har_recorder = HarRecorder(har_path, fallback_url=resolved_start_url) if config.record_har else None

	browser = Browser(
		headless=config.headless,
		viewport={'width': config.viewport_width, 'height': config.viewport_height},
		record_har_path=str(har_path) if config.record_har else None,
	)

	initial_actions = None
	if resolved_start_url:
		initial_actions = [{'navigate': {'url': resolved_start_url}}]

	agent = Agent(
		task=spec.task,
		llm=ChatBrowserUse(),
		browser=browser,
		initial_actions=initial_actions,
		register_new_step_callback=trace_recorder.on_new_step,
		register_done_callback=trace_recorder.on_done,
	)

	start_time = time.time()
	history = asyncio.run(
		_run_agent(
			agent,
			trace_recorder,
			har_recorder,
			resolved_start_url,
			allowed_domains,
			config.max_domain_resets,
			config.max_steps,
		)
	)
	duration = time.time() - start_time

	success = history.is_successful()
	steps = history.number_of_steps()
	final_url = history.urls()[-1] if history.urls() else None
	har_status, har_missing_reason = _resolve_har_status(config.record_har, har_path)

	_write_json(output_dir / 'task.json', spec.model_dump(mode='json'))
	_write_agent_response(output_dir, task_id, spec, history, final_url)
	eval_status, eval_reason = _maybe_run_webarena_verified_eval(config, task_id, output_dir)

	summary = RunSummary(
		task_id=task_id,
		round_id=config.round_id,
		raw_start_url=raw_start_url,
		resolved_start_url=resolved_start_url,
		allowed_domains=allowed_domains,
		success=success,
		steps=steps,
		time_seconds=duration,
		final_url=final_url,
		har_status=har_status,
		har_missing_reason=har_missing_reason,
		eval_status=eval_status,
		eval_reason=eval_reason,
		failure_histogram=trace_recorder.failure_histogram,
	)

	_write_json(output_dir / 'summary.json', summary.model_dump(mode='json'))

	return summary


def _load_task_spec(task_spec: TaskSpec | dict[str, Any] | str) -> TaskSpec:
	if isinstance(task_spec, TaskSpec):
		return task_spec
	if isinstance(task_spec, dict):
		return TaskSpec.model_validate(task_spec)

	path = Path(task_spec)
	if path.exists() and path.is_file():
		with path.open(encoding='utf-8') as handle:
			data = json.load(handle)
		return TaskSpec.model_validate(data)

	return TaskSpec(task=task_spec)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	with path.open('w', encoding='utf-8') as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=True)


def _write_agent_response(
	output_dir: Path,
	task_id: str,
	spec: TaskSpec,
	history: Any,
	final_url: str | None,
) -> None:
	try:
		agent_response = _build_agent_response(task_id, spec, history, final_url)
		_write_json(output_dir / 'agent_response.json', agent_response.model_dump(mode='json'))
	except Exception as exc:
		_write_json(output_dir / 'agent_response.json', {'error': str(exc)})


def _build_agent_response(task_id: str, spec: TaskSpec, history: Any, final_url: str | None) -> AgentResponsePayload:
	final_output = history.final_result()
	if final_output is None:
		extracted = history.extracted_content()
		final_output = extracted[-1] if extracted else ''

	success = history.is_successful()
	status = 'SUCCESS' if success else 'UNKNOWN_ERROR'
	task_type = _infer_task_type(spec)

	retrieved_data = None
	if task_type == 'RETRIEVE':
		retrieved_data = [final_output] if final_output else []

	error_details = None if success else 'Agent did not complete task successfully.'

	return AgentResponsePayload(
		task_id=task_id,
		task_type=task_type,
		status=status,
		retrieved_data=retrieved_data,
		error_details=error_details,
		final_answer=final_output or None,
		final_url=final_url,
		steps=history.number_of_steps(),
		action_summary=history.action_names(),
	)


def _infer_task_type(spec: TaskSpec) -> str:
	if spec.metadata:
		for key in ('task_type', 'expected_task_type', 'main_objective', 'performed_operation'):
			value = spec.metadata.get(key)
			if isinstance(value, str) and value.strip():
				return value.strip().upper()
	return 'RETRIEVE'


def _extract_raw_start_url(spec: TaskSpec) -> str | None:
	if spec.metadata and spec.metadata.get('raw_start_url'):
		return str(spec.metadata.get('raw_start_url'))
	return spec.start_url


def _contains_placeholder(value: str) -> bool:
	return bool(_PLACEHOLDER_PATTERN.search(value))


def _derive_allowed_domains(
	resolved_start_url: str | None,
	env_config: Any,
	metadata: dict[str, Any] | None,
) -> list[str] | None:
	urls: list[str] = []
	if resolved_start_url:
		urls.append(resolved_start_url)

	site_names = metadata.get('sites') if metadata else None
	if site_names and env_config is not None:
		for site in site_names:
			placeholder = _SITE_TO_PLACEHOLDER.get(str(site).lower())
			if placeholder and placeholder in env_config.placeholders:
				urls.append(env_config.placeholders[placeholder])

	hosts = []
	for url in urls:
		parsed = urlparse(url)
		if parsed.hostname:
			hosts.append(parsed.hostname)

	if not hosts:
		return None

	unique_hosts = sorted(set(hosts))
	return [f'http*://{host}' for host in unique_hosts]


def _build_failure_classifier(
	allowed_domains: list[str] | None,
	enable_failure_classification: bool,
):
	def _classifier(error_msg: str | None, state: Any) -> FailureType | None:
		if allowed_domains and state and state.url and not _url_in_allowed_domains(state.url, allowed_domains):
			return FailureType.DOMAIN_DRIFT
		if enable_failure_classification:
			return classify_failure(error_msg, state)
		return None

	return _classifier


async def _run_agent(
	agent: Agent,
	trace_recorder: TraceRecorder,
	har_recorder: HarRecorder | None,
	resolved_start_url: str | None,
	allowed_domains: list[str] | None,
	max_domain_resets: int,
	max_steps: int,
):
	domain_guard = DomainGuard(allowed_domains, resolved_start_url, max_domain_resets)

	async def _on_step_end(inner_agent: Agent) -> None:
		await trace_recorder.on_step_end(inner_agent)
		await domain_guard.handle(inner_agent)

	try:
		if har_recorder is not None:
			await agent.browser_session.start()
			await har_recorder.start(agent.browser_session)

		return await agent.run(max_steps=max_steps, on_step_end=_on_step_end)
	finally:
		if har_recorder is not None:
			try:
				har_recorder.write()
			except Exception:
				pass


def _resolve_har_status(record_har: bool, har_path: Path) -> tuple[str, str | None]:
	if not record_har:
		return 'skipped', 'record_har disabled'
	if har_path.exists():
		return 'ok', None
	return 'missing', 'har file not written'


def _maybe_run_webarena_verified_eval(
	config: RunnerConfig,
	task_id: str,
	output_dir: Path,
) -> tuple[str, str | None]:
	if not config.enable_webarena_verified_eval:
		return 'skipped', 'eval disabled'

	agent_response_path = output_dir / 'agent_response.json'
	har_path = output_dir / 'network.har'
	if not agent_response_path.exists() or not har_path.exists():
		return 'skipped', 'missing agent_response or network.har'

	try:
		from trace_agent.integrations.webarena_verified import evaluate_task

		eval_result = evaluate_task(
			task_id=task_id,
			agent_response_path=agent_response_path,
			har_path=har_path,
			config_path=config.webarena_verified_config_path,
		)
		_write_json(output_dir / 'eval.json', eval_result)
		return 'ok', None
	except Exception as exc:
		_write_json(output_dir / 'eval.json', {'error': str(exc)})
		return 'error', str(exc)


class DomainGuard:
	"""Detect and optionally reset when the agent leaves allowed domains."""

	def __init__(self, allowed_domains: list[str] | None, reset_url: str | None, max_resets: int) -> None:
		self.allowed_domains = allowed_domains or []
		self.reset_url = reset_url
		self.max_resets = max_resets
		self.reset_count = 0

	async def handle(self, agent: Agent) -> None:
		if not self.allowed_domains:
			return
		url = _get_current_url(agent)
		if not url:
			return
		if _url_in_allowed_domains(url, self.allowed_domains):
			return
		if self.reset_url and self.reset_count < self.max_resets:
			self.reset_count += 1
			try:
				await agent.browser_session.navigate_to(self.reset_url)
			except Exception:
				return


def _get_current_url(agent: Agent) -> str | None:
	if agent.history.history:
		item = agent.history.history[-1]
		if item.state and item.state.url:
			return item.state.url
	return agent.history.urls()[-1] if agent.history.urls() else None


def _url_in_allowed_domains(url: str, allowed_domains: list[str]) -> bool:
	if url in ['about:blank', 'chrome://new-tab-page/', 'chrome://new-tab-page', 'chrome://newtab/']:
		return True

	parsed = urlparse(url)
	if parsed.scheme in ['data', 'blob']:
		return True

	host = parsed.hostname
	if not host:
		return False

	for pattern in allowed_domains:
		if _is_url_match(url, host, parsed.scheme, pattern):
			return True
	return False


def _is_url_match(url: str, host: str, scheme: str, pattern: str) -> bool:
	full_url_pattern = f'{scheme}://{host}'
	if '*' in pattern:
		if pattern.startswith('*.'):
			domain_part = pattern[2:]
			if host == domain_part or host.endswith('.' + domain_part):
				if scheme in ['http', 'https']:
					return True
		elif pattern.endswith('/*'):
			if fnmatch.fnmatch(url, pattern):
				return True
		else:
			if fnmatch.fnmatch(full_url_pattern if '://' in pattern else host, pattern):
				return True
	else:
		if '://' in pattern:
			if url.startswith(pattern):
				return True
		else:
			if host.lower() == pattern.lower():
				return True
			if host.lower() == f'www.{pattern.lower()}':
				return True
	return False
