#!/usr/bin/env python3
"""Run a single WebArena-Verified task end-to-end."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

from trace_agent.integrations.webarena_verified_tasks import load_env_config, load_task, resolve_start_url
from trace_agent.runner import RunnerConfig, run_task


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Run one WebArena-Verified task with trace-agent runner.')
	parser.add_argument('--task_id', default='0', help='Task ID from WebArena-Verified dataset')
	parser.add_argument('--env_config', default='config/webarena_verified_env.json', help='Environment config path')
	parser.add_argument('--record_har', action='store_true', help='Enable HAR recording')
	parser.add_argument('--round_id', type=int, default=0, help='Round ID for output directory')
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	task = load_task(args.task_id)

	env_config = load_env_config(args.env_config)
	task = resolve_start_url(task, env_config)

	config = RunnerConfig(
		record_har=args.record_har,
		enable_webarena_verified_eval=True,
		env_config_path=Path(args.env_config),
		webarena_verified_config_path=Path(args.env_config),
		round_id=args.round_id,
	)
	summary = run_task(task, config)

	run_dir = config.output_root / summary.task_id / f'round_{config.round_id}'
	expected_files = [
		run_dir / 'trace.jsonl',
		run_dir / 'summary.json',
		run_dir / 'agent_response.json',
		run_dir / 'network.har',
		run_dir / 'eval.json',
	]

	print(f'Run directory: {run_dir}')
	missing = []
	for path in expected_files:
		status = 'OK' if path.exists() else 'MISSING'
		print(f'{path}: {status}')
		if status == 'MISSING':
			missing.append(path)

	if missing:
		print(f'Missing required files: {missing}')
		sys.exit(1)


if __name__ == '__main__':
	main()
