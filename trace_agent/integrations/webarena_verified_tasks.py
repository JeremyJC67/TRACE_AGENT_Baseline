"""Task loader for WebArena-Verified dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


DATASET_RELATIVE_PATH = Path('third_party/webarena_verified/assets/dataset/webarena-verified.json')
_PLACEHOLDER_PATTERN = re.compile(r'__[^_]+__')
_ENV_KEY_TO_PLACEHOLDER = {
	'SHOPPING_ADMIN_URL': '__SHOPPING_ADMIN__',
	'SHOPPING_URL': '__SHOPPING__',
	'GITLAB_URL': '__GITLAB__',
	'REDDIT_URL': '__REDDIT__',
	'WIKIPEDIA_URL': '__WIKIPEDIA__',
	'MAP_URL': '__MAP__',
}


class EnvConfig(BaseModel):
	"""Normalized environment config mapping placeholders to URLs."""

	placeholders: dict[str, str] = Field(default_factory=dict)


class StartUrlResolution(BaseModel):
	"""Resolved start URL details."""

	raw_start_url: str | None
	resolved_start_url: str | None
	unresolved_placeholders: list[str] = Field(default_factory=list)


def load_task(task_id: str) -> dict[str, Any]:
	"""Load a task from WebArena-Verified dataset by task_id."""
	task_id_int = _coerce_task_id(task_id)
	dataset_path = _resolve_dataset_path()

	with dataset_path.open(encoding='utf-8') as handle:
		data = json.load(handle)

	if not isinstance(data, list):
		raise ValueError('Dataset JSON is not a list.')

	for item in data:
		if item.get('task_id') == task_id_int:
			return _to_task_spec(item)

	raise KeyError(f'Task not found: {task_id}')


def _resolve_dataset_path() -> Path:
	repo_root = Path(__file__).resolve().parents[2]
	dataset_path = repo_root / DATASET_RELATIVE_PATH
	if not dataset_path.exists():
		raise FileNotFoundError(f'WebArena-Verified dataset not found: {dataset_path}')
	return dataset_path


def _coerce_task_id(task_id: str) -> int:
	if not task_id:
		raise ValueError('task_id is required')
	try:
		return int(task_id)
	except ValueError as exc:
		raise ValueError(f'task_id must be numeric, got: {task_id!r}') from exc


def _to_task_spec(item: dict[str, Any]) -> dict[str, Any]:
	start_urls = item.get('start_urls') or []
	start_url = start_urls[0] if start_urls else None
	return {
		'task_id': str(item.get('task_id')),
		'task': item.get('intent') or item.get('instruction') or '',
		'start_url': start_url,
		'metadata': {
			'sites': item.get('sites'),
			'intent_template_id': item.get('intent_template_id'),
			'revision': item.get('revision'),
			'expected_task_type': _extract_expected_task_type(item),
		},
	}


def _extract_expected_task_type(item: dict[str, Any]) -> str | None:
	eval_cfgs = item.get('eval') or []
	if not eval_cfgs:
		return None
	expected = eval_cfgs[0].get('expected') if isinstance(eval_cfgs[0], dict) else None
	if not isinstance(expected, dict):
		return None
	task_type = expected.get('task_type') or expected.get('performed_operation')
	if isinstance(task_type, str) and task_type.strip():
		return task_type.strip().upper()
	return None


def load_env_config(path: str | Path) -> EnvConfig:
	"""Load environment config and normalize placeholder mappings."""
	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f'Env config not found: {config_path}')

	raw = json.loads(config_path.read_text(encoding='utf-8'))
	placeholders: dict[str, str] = {}

	if isinstance(raw, dict) and 'environments' in raw:
		envs = raw.get('environments') or {}
		for placeholder, env in envs.items():
			if not isinstance(env, dict):
				continue
			urls = env.get('urls') or []
			active_idx = env.get('active_url_idx')
			if urls:
				try:
					url = urls[int(active_idx)] if active_idx is not None else urls[0]
				except (ValueError, IndexError):
					url = urls[0]
				placeholders[str(placeholder)] = str(url).rstrip('/')
	elif isinstance(raw, dict):
		for key, value in raw.items():
			if key in _ENV_KEY_TO_PLACEHOLDER and value:
				placeholders[_ENV_KEY_TO_PLACEHOLDER[key]] = str(value).rstrip('/')
			elif key.startswith('__') and key.endswith('__') and value:
				placeholders[key] = str(value).rstrip('/')

	return EnvConfig(placeholders=placeholders)


def resolve_start_url(task: dict[str, Any], env_config: EnvConfig) -> dict[str, Any]:
	"""Resolve placeholders in task.start_url using environment config."""
	raw_start_url = task.get('start_url')
	resolution = _resolve_start_url_value(raw_start_url, env_config.placeholders)
	if resolution.unresolved_placeholders:
		raise ValueError(
			f'Unresolved placeholders in start_url: {resolution.unresolved_placeholders}. '
			f'Provide mappings in env_config.'
		)

	updated = dict(task)
	updated['start_url'] = resolution.resolved_start_url
	metadata = dict(task.get('metadata') or {})
	metadata['raw_start_url'] = resolution.raw_start_url
	updated['metadata'] = metadata
	return updated


def _resolve_start_url_value(start_url: str | None, placeholders: dict[str, str]) -> StartUrlResolution:
	if not start_url:
		return StartUrlResolution(raw_start_url=start_url, resolved_start_url=start_url, unresolved_placeholders=[])

	resolved = start_url
	for placeholder, url in placeholders.items():
		resolved = resolved.replace(placeholder, url)

	unresolved = [match.group(0) for match in _PLACEHOLDER_PATTERN.finditer(resolved)]
	return StartUrlResolution(
		raw_start_url=start_url,
		resolved_start_url=resolved,
		unresolved_placeholders=sorted(set(unresolved)),
	)
