"""Durable reservation and exactly-once terminal resource accounting."""

import uuid
from harness.state_store import transaction

TRAIN = {"run_full_validation"}
PROBE = {"run_smoke_test", "run_probe"}


def reserve(path, session, action, backend, budget, deadline=None):
    token = uuid.uuid4().hex
    with transaction(path, {"reservations": {}}) as ledger:
        import json

        state_path = path.parent / "competition_state.json"
        if state_path.is_file() and json.loads(state_path.read_text()).get(
            "gates", {}
        ).get("freeze_locked"):
            raise ValueError("Competition is frozen")
        rows = [
            r for r in ledger["reservations"].values() if r["session_id"] == session
        ]
        category = (
            "training"
            if action in TRAIN
            else "probe" if action in PROBE else "candidate"
        )
        limit = (
            budget.get("max_experiments")
            if category == "training"
            else (
                budget.get("max_probes")
                if category == "probe"
                else budget.get("max_candidate_operations")
            )
        )
        if limit is not None and sum(r["category"] == category for r in rows) >= int(
            limit
        ):
            raise ValueError("Resource reservation budget exhausted")
        if (
            backend != "local"
            and budget.get("max_cloud_runs") is not None
            and sum(r["backend"] != "local" for r in rows)
            >= int(budget["max_cloud_runs"])
        ):
            raise ValueError("Cloud reservation budget exhausted")
        if deadline is not None:
            allocated = sum(
                float(r.get("wall_seconds") or r.get("reserved_seconds") or 0)
                for r in rows
            )
            remaining = float(budget["max_wall_seconds"]) - allocated
            if remaining <= 0:
                raise ValueError("Wall-time reservation budget exhausted")
            deadline = min(deadline, remaining)
        ledger["reservations"][token] = {
            "session_id": session,
            "category": category,
            "backend": backend,
            "status": "RESERVED",
            "reserved_seconds": deadline,
            "wall_seconds": None,
            "gpu_hours": None,
        }
    return token, deadline


def terminal(
    path, token, status, wall_seconds, gpu_count=None, allocation_seconds=None
):
    with transaction(path, {"reservations": {}}) as ledger:
        row = ledger["reservations"][token]
        if row["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return False
        row.update(
            status=status,
            wall_seconds=wall_seconds,
            gpu_hours=(
                None
                if gpu_count is None or allocation_seconds is None
                else gpu_count * allocation_seconds / 3600
            ),
            usage_source=(
                "provider_allocation" if gpu_count is not None else "process_wall"
            ),
        )
    return True
