"""Trace recorder for minimal-intrusion agent instrumentation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from browser_use.agent.views import AgentOutput
from browser_use.browser.views import BrowserStateHistory, BrowserStateSummary

from trace_agent.failure import FailureType, classify_failure

if TYPE_CHECKING:
	from browser_use.agent.service import Agent


class ActionRecord(BaseModel):
	"""A single action chosen by the agent."""

	action_type: str
	action_args: dict[str, Any] | None = None


class TraceStep(BaseModel):
	"""Schema for one step in the execution trace."""

	model_config = ConfigDict(extra='allow')

	step_id: int
	timestamp: str
	url: str | None = None
	title: str | None = None
	action_type: str | None = None
	action_args: dict[str, Any] | None = None
	result_status: str | None = None
	error_msg: str | None = None
	failure_type: FailureType | None = None
	latency_ms: int | None = None
	screenshot_path: str | None = None
	dom_hash: str | None = None
	actions: list[ActionRecord] | None = None


class TraceRecorder:
	"""Write JSONL traces for each agent step."""

	def __init__(
		self,
		output_dir: Path,
		failure_classifier: Callable[[str | None, BrowserStateHistory | None], FailureType | None] | None = None,
	) -> None:
		self.output_dir = output_dir
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.trace_path = self.output_dir / 'trace.jsonl'
		self._pending: dict[int, TraceStep] = {}
		self._failure_histogram: dict[str, int] = {}
		self._failure_classifier = failure_classifier or classify_failure

	async def on_new_step(self, state: BrowserStateSummary, model_output: AgentOutput, step_id: int) -> None:
		"""Capture pre-action state + model output."""
		actions = _extract_actions(model_output)
		first_action = actions[0] if actions else None
		trace = TraceStep(
			step_id=step_id,
			timestamp=_utc_now(),
			url=state.url,
			title=state.title,
			action_type=first_action.action_type if first_action else None,
			action_args=first_action.action_args if first_action else None,
			actions=actions or None,
			dom_hash=_compute_dom_hash(state),
		)
		self._pending[step_id] = trace

	async def on_step_end(self, agent: Agent) -> None:
		"""Finalize a trace step after actions execute."""
		history_item = agent.history.history[-1] if agent.history.history else None
		if not history_item:
			return

		step_id = _get_step_id(history_item.metadata.step_number if history_item.metadata else None, agent)
		trace = self._pending.pop(step_id, TraceStep(step_id=step_id, timestamp=_utc_now()))

		result_status, error_msg = _summarize_results(history_item.result)
		failure_type = self._failure_classifier(error_msg, history_item.state) if error_msg else None

		latency_ms = None
		if history_item.metadata:
			latency_ms = int(history_item.metadata.duration_seconds * 1000)
		elif getattr(agent, 'step_start_time', None):
			latency_ms = int((time.time() - agent.step_start_time) * 1000)

		trace.result_status = result_status
		trace.error_msg = error_msg
		trace.failure_type = failure_type
		trace.latency_ms = latency_ms
		trace.screenshot_path = history_item.state.screenshot_path

		self._record_failure(failure_type)
		_write_jsonl(self.trace_path, trace.model_dump(mode='json'))

	async def on_done(self, _history: Any) -> None:
		"""Hook point for done; currently no-op."""
		return None

	@property
	def failure_histogram(self) -> dict[str, int]:
		return dict(self._failure_histogram)

	def _record_failure(self, failure_type: FailureType | None) -> None:
		if not failure_type:
			return
		key = failure_type.value
		self._failure_histogram[key] = self._failure_histogram.get(key, 0) + 1


def _extract_actions(model_output: AgentOutput) -> list[ActionRecord]:
	actions: list[ActionRecord] = []
	if not model_output or not model_output.action:
		return actions

	for action in model_output.action:
		payload = action.model_dump(exclude_none=True, mode='json')
		if not payload:
			continue
		action_type = list(payload.keys())[0]
		action_args = payload.get(action_type)
		actions.append(ActionRecord(action_type=action_type, action_args=action_args))
	return actions


def _summarize_results(results: list[Any]) -> tuple[str | None, str | None]:
	if not results:
		return None, None

	error_msg = None
	for result in results:
		if getattr(result, 'error', None):
			error_msg = result.error
			break

	last_result = results[-1]
	if error_msg:
		return 'error', error_msg
	if getattr(last_result, 'is_done', False):
		if getattr(last_result, 'success', None) is True:
			return 'done_success', None
		if getattr(last_result, 'success', None) is False:
			return 'done_failure', None
	return 'ok', None


def _compute_dom_hash(state: BrowserStateSummary) -> str | None:
	try:
		dom_text = state.dom_state.llm_representation()
	except Exception:
		return None
	return hashlib.sha256(dom_text.encode('utf-8')).hexdigest()


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
	with path.open('a', encoding='utf-8') as handle:
		handle.write(json.dumps(record, ensure_ascii=True) + '\n')


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _get_step_id(step_number: int | None, agent: Agent) -> int:
	if step_number is not None:
		return step_number
	return max(1, getattr(agent.state, 'n_steps', 1) - 1)
