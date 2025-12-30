#!/usr/bin/env python3
"""Smoke test for WebArena-Verified integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_webarena_verified_on_path(repo_root: Path) -> None:
	src_path = repo_root / 'third_party' / 'webarena_verified' / 'src'
	if not src_path.exists():
		raise FileNotFoundError(f'WebArena-Verified src not found: {src_path}')
	sys.path.insert(0, str(src_path))


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	try:
		_ensure_webarena_verified_on_path(repo_root)
	except FileNotFoundError as exc:
		raise SystemExit(f'[smoke] {exc}') from exc

	try:
		from webarena_verified.api import WebArenaVerified  # noqa: F401
		from webarena_verified.types.config import WebArenaVerifiedConfig  # noqa: F401
	except ImportError as exc:
		raise SystemExit(
			'[smoke] Failed to import webarena_verified. Ensure the submodule is cloned and PYTHONPATH includes '
			'third_party/webarena_verified/src.'
		) from exc

	dataset_path = repo_root / 'third_party' / 'webarena_verified' / 'assets' / 'dataset' / 'webarena-verified.json'
	if not dataset_path.exists():
		raise SystemExit(f'[smoke] Dataset not found: {dataset_path}')

	with dataset_path.open(encoding='utf-8') as handle:
		data = json.load(handle)

	if not isinstance(data, list) or not data:
		raise SystemExit('[smoke] Dataset JSON is empty or not a list.')

	sample_task = data[0]
	task_id = sample_task.get('task_id', 'unknown')
	print(f'[smoke] Loaded {len(data)} tasks. sample task_id={task_id}')


if __name__ == '__main__':
	main()
