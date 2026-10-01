"""Resolve epoch-triggered runtime-schedule stages.

A :class:`RuntimeScheduleConfig` pairs ``milestones`` with ``states``. Given the
current epoch, :func:`resolve_runtime_schedule_stage` returns the stage that has
just become active (i.e. when ``epoch`` exactly equals a milestone), so the
runner applies each transition once.
"""

from typing import Any, Dict, Optional, Tuple


def resolve_runtime_schedule_stage(schedule, epoch: int) -> Optional[Tuple[int, Dict[str, Any]]]:
    """Return ``(stage_index, state)`` if a milestone fires at ``epoch``, else None."""
    if schedule is None:
        return None
    milestones = list(getattr(schedule, "milestones", []) or [])
    states = list(getattr(schedule, "states", []) or [])
    for idx, milestone in enumerate(milestones):
        if int(milestone) == int(epoch) and idx < len(states):
            return idx, states[idx]
    return None
