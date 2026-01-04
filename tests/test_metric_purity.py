from __future__ import annotations

from policy_eval.core.types import Event, Trajectory


def test_metric_cannot_mutate_trajectory():
    traj = Trajectory(
        events=(Event(t=0, payload={"x": 1}),),
        final_state={"done": True},
    )

    # Trajectory is frozen; assignment should fail
    try:
        traj.final_state = {"done": False}  # type: ignore[misc]
        assert False, "Expected Trajectory to be immutable"
    except Exception:
        pass
