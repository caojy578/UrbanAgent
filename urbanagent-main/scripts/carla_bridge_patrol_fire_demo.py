"""Run UrbanMultiAgentSystem.run_patrol_fire_response against CarlaBridge.

Closed loop: 无火情 -> UAV 巡逻 -> state_snapshot 出现新 fire incident ->
event_log 提示 -> UrbanAgent 火情调度 -> 灭火完成后 UAV/UGV 返航.

NOTE: Do not supply --fallback-incident-id unless you want to bypass the patrol
phase. The patrol loop only triggers when the initial CarlaBridge snapshot has
no open fire incident.

Example:
    python scripts/carla_bridge_patrol_fire_demo.py --url http://127.0.0.1:5000 --no-llm
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from urbanagent import CarlaBridgeSandboxClient, UrbanMultiAgentSystem
from urbanagent.types import Coordinate, Incident


def _parse_waypoint(text: str) -> Coordinate:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"waypoint must be x,y,z (got {text!r})"
        )
    try:
        return Coordinate(float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _print_batch_outcome(label: str, outcome) -> None:
    if outcome is None:
        print(f"[{label}] outcome=None")
        return
    print(
        f"[{label}] batch_id={outcome.batch_id} "
        f"criteria_satisfied={outcome.criteria_satisfied} "
        f"steps={len(outcome.per_step_results)}"
    )
    for step in outcome.per_step_results:
        print(
            f"  - {step.action.kind} target={step.action.target_id} "
            f"status={step.status} message={step.message}"
        )
    for note in outcome.notes:
        print(f"  note: {note}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="CarlaBridge URL.")
    parser.add_argument("--namespace", default="/agent", help="Socket.IO namespace.")
    parser.add_argument("--dotenv-path", default=".env", help="LLM .env path.")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM.")

    parser.add_argument(
        "--patrol-waypoint",
        type=_parse_waypoint,
        action="append",
        default=None,
        help=(
            "Patrol waypoint in 'x,y,z' (meters, CARLA frame). Pass multiple "
            "times to define a polyline. When omitted, each UAV gets an auto "
            "square around its current pose."
        ),
    )
    parser.add_argument("--patrol-altitude", type=float, default=60.0)
    parser.add_argument("--patrol-radius", type=float, default=80.0)
    parser.add_argument(
        "--max-patrol-drones",
        type=int,
        default=3,
        help="Cap of UAVs to send on the initial patrol batch.",
    )

    parser.add_argument(
        "--detection-poll-interval",
        type=float,
        default=0.5,
        help="Seconds between two state_snapshot polls when waiting for fire.",
    )
    parser.add_argument(
        "--max-detection-rounds",
        type=int,
        default=120,
        help="Max polls before giving up (default: 60s at 0.5s interval).",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="Skip the post-response RTL batch (debug only).",
    )

    parser.add_argument(
        "--fallback-incident-id",
        default=None,
        help=(
            "If set, inject a default fire incident into the CarlaBridge "
            "snapshot when it has no incidents. This BYPASSES the patrol "
            "phase. Only use to debug the dispatch leg in isolation."
        ),
    )
    parser.add_argument("--fallback-x", type=float, default=0.0)
    parser.add_argument("--fallback-y", type=float, default=0.0)
    parser.add_argument("--fallback-z", type=float, default=0.0)

    parser.add_argument("--command-timeout", type=float, default=180.0)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)

    args = parser.parse_args()

    default_incidents: list[Incident] = []
    if args.fallback_incident_id:
        default_incidents.append(
            Incident(
                id=args.fallback_incident_id,
                kind="fire",
                severity="high",
                position=Coordinate(args.fallback_x, args.fallback_y, args.fallback_z),
                description="Fallback incident supplied by patrol-fire demo.",
            )
        )

    sandbox = CarlaBridgeSandboxClient(
        args.url,
        namespace=args.namespace,
        default_incidents=default_incidents,
        command_timeout=args.command_timeout,
        ack_timeout=args.ack_timeout,
        state_timeout=args.state_timeout,
    )
    agent = UrbanMultiAgentSystem(
        sandbox=sandbox,
        dotenv_path=args.dotenv_path,
        use_llm=not args.no_llm,
        use_llm_batch_rerank=not args.no_llm,
    )

    try:
        result = await agent.run_patrol_fire_response(
            patrol_waypoints=args.patrol_waypoint,
            patrol_altitude=args.patrol_altitude,
            patrol_radius=args.patrol_radius,
            max_patrol_drones=args.max_patrol_drones,
            detection_poll_interval_s=args.detection_poll_interval,
            max_detection_rounds=args.max_detection_rounds,
            return_after_response=not args.no_return,
        )
    finally:
        await sandbox.close()

    print("=" * 72)
    print("final_report:", result.final_report)
    print("detected_incident_id:", result.detected_incident_id)
    print("detection_notes:")
    for note in result.detection_notes:
        print(f"  - {note}")

    _print_batch_outcome("PATROL", result.patrol_outcome)

    if result.response is not None:
        print(
            f"[RESPONSE] llm_used={result.response.llm_used} "
            f"skipped_reason={result.response.skipped_reason!r}"
        )
        if result.response.committed is not None:
            print(
                f"[RESPONSE] committed batch_id={result.response.committed.batch_id} "
                f"actions={len(result.response.committed.actions)}"
            )
        _print_batch_outcome("RESPONSE", result.response.batch_outcome)
        if result.response.final_report:
            print("[RESPONSE] final_report:", result.response.final_report)

    _print_batch_outcome("RETURN", result.return_outcome)

    response_ok = (
        result.response is not None
        and result.response.batch_outcome is not None
        and result.response.batch_outcome.criteria_satisfied
    )
    if result.detected_incident_id is None:
        return 2  # no fire detected within window
    return 0 if response_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
