"""Trace Agent augmentation package."""

from trace_agent.failure import FailureType, classify_failure
from trace_agent.runner import RunnerConfig, TaskSpec, run_task
from trace_agent.trace_recorder import TraceRecorder

__all__ = [
	'FailureType',
	'RunnerConfig',
	'TaskSpec',
	'TraceRecorder',
	'classify_failure',
	'run_task',
]
