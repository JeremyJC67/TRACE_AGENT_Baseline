"""Integration adapter for WebArena-Verified evaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


def evaluate_task(
	task_id: str,
	agent_response_path: Path,
	har_path: Path,
	config_path: Optional[Path] = None,
) -> dict[str, Any]:
	"""Evaluate a task using WebArena-Verified.

	TODO: Map trace_agent runner outputs to WebArena-Verified agent_response + HAR formats.
	"""
	_task_id = _validate_task_id(task_id)
	agent_response_path = _validate_file(agent_response_path, 'agent_response_path')
	har_path = _validate_file(har_path, 'har_path')
	if config_path is not None:
		config_path = _validate_file(config_path, 'config_path')

	_ensure_webarena_verified_on_path()

	try:
		from webarena_verified.api import WebArenaVerified
	except ImportError as exc:
		raise RuntimeError(
			'Failed to import webarena_verified. Ensure the submodule is cloned and '
			'third_party/webarena_verified/src is on PYTHONPATH.'
		) from exc

	wa = WebArenaVerified(config=config_path)
	result = wa.evaluate_task(
		task_id=_task_id,
		agent_response=agent_response_path,
		network_trace=har_path,
	)
	return result.model_dump(mode='json')


def _ensure_webarena_verified_on_path() -> None:
	repo_root = Path(__file__).resolve().parents[2]
	src_path = repo_root / 'third_party' / 'webarena_verified' / 'src'
	if not src_path.exists():
		raise RuntimeError(f'WebArena-Verified src not found: {src_path}')
	if str(src_path) not in sys.path:
		sys.path.insert(0, str(src_path))


def _validate_task_id(task_id: str) -> int:
	if not task_id:
		raise ValueError('task_id is required')
	try:
		return int(task_id)
	except ValueError as exc:
		raise ValueError(f'task_id must be numeric, got: {task_id!r}') from exc


def _validate_file(path: Path, label: str) -> Path:
	if not isinstance(path, Path):
		raise TypeError(f'{label} must be a Path')
	if not path.exists():
		raise FileNotFoundError(f'{label} not found: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'{label} is not a file: {path}')
	return path
