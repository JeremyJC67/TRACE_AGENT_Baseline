"""Single-round runner for baseline agent with tracing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from browser_use import Agent, Browser, ChatBrowserUse

from trace_agent.trace_recorder import TraceRecorder


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


class RunSummary(BaseModel):
	"""Summary of one run for analysis."""

	task_id: str
	round_id: int
	success: bool | None
	steps: int
	time_seconds: float | None
	final_url: str | None
	failure_histogram: dict[str, int] = Field(default_factory=dict)


def run_task(task_spec: TaskSpec | dict[str, Any] | str, config: RunnerConfig | None = None) -> RunSummary:
	"""Run a baseline agent and write trace + summary to disk."""
	config = config or RunnerConfig()
	spec = _load_task_spec(task_spec)
	task_id = spec.task_id or f'task_{uuid4().hex[:8]}'

	output_dir = config.output_root / task_id / f'round_{config.round_id}'
	output_dir.mkdir(parents=True, exist_ok=True)

	trace_recorder = TraceRecorder(
		output_dir=output_dir,
		failure_classifier=None if config.enable_failure_classification else _noop_failure_classifier,
	)

	browser = Browser(
		headless=config.headless,
		viewport={'width': config.viewport_width, 'height': config.viewport_height},
	)

	initial_actions = None
	if spec.start_url:
		initial_actions = [{'navigate': {'url': spec.start_url}}]

	agent = Agent(
		task=spec.task,
		llm=ChatBrowserUse(),
		browser=browser,
		initial_actions=initial_actions,
		register_new_step_callback=trace_recorder.on_new_step,
		register_done_callback=trace_recorder.on_done,
	)

	start_time = time.time()
	history = agent.run_sync(max_steps=config.max_steps, on_step_end=trace_recorder.on_step_end)
	duration = time.time() - start_time

	success = history.is_successful()
	steps = history.number_of_steps()
	final_url = history.urls()[-1] if history.urls() else None

	summary = RunSummary(
		task_id=task_id,
		round_id=config.round_id,
		success=success,
		steps=steps,
		time_seconds=duration,
		final_url=final_url,
		failure_histogram=trace_recorder.failure_histogram,
	)

	_write_json(output_dir / 'summary.json', summary.model_dump(mode='json'))
	_write_json(output_dir / 'task.json', spec.model_dump(mode='json'))

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


def _noop_failure_classifier(_error_msg: str | None, _state: Any) -> None:
	return None
