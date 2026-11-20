"""Local PiPER vision setup, read-only diagnosis, explicit own registration/export."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.goal_verifier.contracts import NAME, need
from plugins.goal_verifier.configuration import template
from plugins.goal_verifier.diagnostics import check, http
from plugins.goal_verifier.server import loads, rpc


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def replay(record):
    from plugins.goal_verifier.engine import Engine

    need(record["schema"] == "motus.piper-goal-replay/1", "REPLAY_SCHEMA")
    cfg = deepcopy(record["config"])
    before = deepcopy(record["before"])
    after = deepcopy(record["after"])
    for binding in cfg["sources"].values():
        binding["allow_test_data"] = True
    for binding in cfg["sources"].values():
        binding.update(
            accept_external_observations=True,
            external_observation_record="OFFLINE-REPLAY-ONLY",
        )
    before["provenance"] = after["provenance"] = "replay"

    class RecordedClock:
        value = before["captured_at"] + 1e-6

        def __call__(self):
            return self.value

    clock = RecordedClock()
    engine = Engine(cfg, wall=clock)
    s = engine.call(
        "start_session",
        dict(
            request_id="start",
            task_id="OFFLINE-REPLAY",
            source_id=before["source_id"],
            selector=record["selector"],
            goal=record["goal"],
        ),
    )
    identity = {k: s[k] for k in ("session_id", "session_token")}
    engine.call(
        "capture_before", dict(**identity, request_id="before", observation=before)
    )
    clock.value = record["operation_completed_at"]
    marker = engine.call("mark_operation_complete", dict(**identity, request_id="mark"))
    clock.value = after["captured_at"] + 1e-6
    out = engine.call(
        "verify_after",
        dict(
            **identity,
            request_id="after",
            marker_id=marker["marker_id"],
            observation=after,
        ),
    )
    return dict(
        scope="OFFLINE_RECORDED_TIMELINE",
        timestamps_preserved=True,
        current_physical_state_verified=False,
        result=out,
        evidence=engine.call("export_session", identity),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("template")
    s.add_argument("--mode", choices=["color", "vop", "normalized"], default="color")
    s.add_argument("--output", required=True)
    s = sub.add_parser("doctor")
    s.add_argument("--config", required=True)
    s.add_argument("--core-url")
    s.add_argument("--mcp-url")
    s.add_argument("--ros", action="store_true")
    s.add_argument("--sample", action="store_true")
    s.add_argument("--token-file")
    s.add_argument("--ca")
    s.add_argument("--output", required=True)
    s = sub.add_parser("fit-plane")
    s.add_argument("--measurements", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("replay")
    s.add_argument("--record", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("inspect-vop")
    s.add_argument("--input", required=True)
    s.add_argument("--frame-record")
    s.add_argument("--config", required=True)
    s.add_argument("--source-id", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("register")
    s.add_argument("--core-url", required=True)
    s.add_argument("--mcp-url", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--token-file")
    s.add_argument("--ca")
    s.add_argument("--output", required=True)
    s = sub.add_parser("unregister")
    s.add_argument("--receipt", required=True)
    s.add_argument("--token-file")
    s.add_argument("--ca")
    s.add_argument("--output", required=True)
    s = sub.add_parser("export")
    s.add_argument("--mcp-url", required=True)
    s.add_argument("--session-id", required=True)
    s.add_argument("--session-token-file", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("package")
    s.add_argument("--output", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--payload", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("rollback")
    s.add_argument("--stage", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--remove-owned-stage", action="store_true")
    a = p.parse_args()
    try:
        if a.cmd == "template":
            result = template(a.mode)
            write(a.output, result)
        elif a.cmd == "doctor":
            result = check(
                loads(Path(a.config).read_text()),
                core=a.core_url,
                mcp=a.mcp_url,
                ros=a.ros,
                sample=a.sample,
                token_file=a.token_file,
                ca=a.ca,
            )
            write(a.output, result)
        elif a.cmd == "fit-plane":
            from plugins.goal_verifier.calibration import fit

            result = fit(loads(Path(a.measurements).read_text()))
            write(a.output, result)
        elif a.cmd == "replay":
            result = replay(loads(Path(a.record).read_text()))
            write(a.output, result)
        elif a.cmd == "inspect-vop":
            from plugins.goal_verifier.sources import from_vop
            from plugins.goal_verifier.contracts import error_result

            try:
                cfg = loads(Path(a.config).read_text())
                frame = (
                    loads(Path(a.frame_record).read_text()) if a.frame_record else None
                )
                result = from_vop(
                    loads(Path(a.input).read_text()), frame, cfg["sources"][a.source_id]
                )
            except Exception as exc:
                result = error_result(exc)
            write(a.output, result)
        elif a.cmd == "register":
            # Explicit action, never part of doctor. Reject existing entries
            # rather than exploiting Core's URL/name upsert to overwrite one.
            need(not Path(a.output).exists(), "RECEIPT_PATH_ALREADY_EXISTS")
            url = a.core_url.rstrip("/")
            listing = http(url + "/api/mcp", token_file=a.token_file, ca=a.ca)["data"]
            need(
                not any(
                    m.get("name") == a.name
                    or m.get("url", "").rstrip("/") == a.mcp_url.rstrip("/")
                    for m in listing
                ),
                "EXISTING_MCP_PRESERVED",
            )
            need(
                any(t["name"] == NAME for t in rpc(a.mcp_url, "tools/list")["tools"]),
                "CARD_NOT_FOUND",
            )
            response = http(
                url + "/api/mcp",
                dict(name=a.name, url=a.mcp_url, category="perception"),
                token_file=a.token_file,
                ca=a.ca,
            )
            need(response.get("code") == 200, "REGISTRATION_FAILED")
            result = dict(
                schema="motus.piper-goal-registration/1",
                core_url=url,
                mcp_url=a.mcp_url,
                name=a.name,
                id=response["data"]["id"],
                uid=os.getuid(),
            )
            write(a.output, result)
        elif a.cmd == "unregister":
            need(not Path(a.output).exists(), "RECEIPT_PATH_ALREADY_EXISTS")
            record = loads(Path(a.receipt).read_text())
            need(
                record["schema"] == "motus.piper-goal-registration/1"
                and record["uid"] == os.getuid(),
                "OWN_REGISTRATION_REQUIRED",
            )
            listing = http(
                record["core_url"] + "/api/mcp", token_file=a.token_file, ca=a.ca
            )["data"]
            selected = next((m for m in listing if m["id"] == record["id"]), None)
            need(
                selected
                and selected["url"] == record["mcp_url"]
                and selected["name"] == record["name"],
                "REGISTRATION_CHANGED_PRESERVE",
            )
            result = http(
                record["core_url"] + "/api/mcp/" + record["id"],
                method="DELETE",
                token_file=a.token_file,
                ca=a.ca,
            )
            write(a.output, result)
        elif a.cmd == "export":
            value = rpc(
                a.mcp_url,
                "tools/call",
                dict(
                    name=NAME,
                    arguments=dict(
                        action="export_session",
                        session_id=a.session_id,
                        session_token=Path(a.session_token_file).read_text().strip(),
                    ),
                ),
            )
            result = loads(value["content"][0]["text"])
            need(result.get("state") != "error", "EXPORT_REFUSED")
            write(a.output, result)
        elif a.cmd == "package":
            from plugins.goal_verifier.packaging import build

            result = build(a.output)
        elif a.cmd == "stage":
            from plugins.goal_verifier.packaging import stage

            result = stage(a.payload, a.output)
        else:
            from plugins.goal_verifier.packaging import rollback

            result = rollback(a.stage, a.run_id, remove=a.remove_owned_stage)
        print(json.dumps(result, indent=2, allow_nan=False))
        if a.cmd == "doctor" and result["state"] == "BLOCKED":
            raise SystemExit(2)
        if a.cmd == "inspect-vop" and result.get("state") == "error":
            raise SystemExit(2)
    except Exception as exc:
        p.exit(
            2,
            "BLOCKED: "
            + getattr(exc, "code", type(exc).__name__)
            + ": "
            + str(exc)
            + "\n",
        )


if __name__ == "__main__":
    main()
