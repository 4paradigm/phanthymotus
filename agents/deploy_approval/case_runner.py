"""Fixed safe case runner for Deploy Approval (final alignment).

Only allowlisted, Controller-owned fixed cases. No subprocess, no shell,
no SSH, no user-specified executable input.

The case runner accepts a dict (not a Deployment model) and accesses
only dictionary keys.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


# Fixed case registry: Controller-owned, never user-supplied.
# Each case has:
#   - id: unique identifier
#   - target: which target it applies to (perception, actucore, driver)
#   - description: human-readable description
#   - timeout: maximum execution time in seconds
#
# Cases use only the existing Agent Core HTTP API (driver_status, MCP ping).
# No subprocess, no shell, no SSH, no Docker socket.

_FIXED_CASES: list[dict[str, Any]] = [
    {
        "id": "perception-health-check",
        "target": "perception",
        "description": "Verify the expected deployed image is bound and MCP is responsive",
        "requires_mcp": True,
        "timeout": 30,
    },
    {
        "id": "actucore-health-check",
        "target": "actucore",
        "description": "Verify the expected deployed image is bound and MCP is responsive",
        "requires_mcp": True,
        "timeout": 30,
    },
    {
        "id": "driver-health-check",
        "target": "driver",
        "description": "Verify the expected deployed image is bound and MCP is responsive",
        "requires_mcp": True,
        "timeout": 30,
    },
]


class CaseRunner:
    """Runner for fixed safe cases.

    Only executes Controller-owned allowlisted cases. Never accepts
    user-supplied scripts, commands, or parameters.

    The deployment parameter is a dict (not a Deployment model) with keys:
      - component_id
      - target
      - variant
      - driver_path
      - node_id
      - node_host
      - image_ref
      - machine_alias (optional)
    """

    def __init__(self, config: Config, core=None, driver_id: str = ""):
        self.config = config
        self._registry = list(_FIXED_CASES)
        self._core = core
        self._driver_id = driver_id

    def select_case(self, target: str, variant: str,
                    node_id: str) -> str | None:
        """Select a compatible fixed case for the given target.

        Returns the case id, or None if no compatible case is available.
        """
        for case in self._registry:
            if case.get("target") == target:
                return case["id"]
        return None

    async def run_case(self, case_id: str,
                       deployment: dict) -> dict[str, Any]:
        """Run a fixed case and return structured results.

        deployment is a dict, not a Deployment model.
        Uses dict keys only (no attribute access).

        Returns:
            dict with keys:
                - passed: bool
                - case_id: str
                - logs: list[str]
                - error: str (empty if passed)
        """
        case = self._find_case(case_id)
        if case is None:
            return {
                "passed": False,
                "case_id": case_id,
                "logs": [],
                "error": f"Unknown case: {case_id}",
            }

        logs: list[str] = []
        timeout = case.get("timeout", 30)

        if not isinstance(deployment, dict):
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment must be a dict",
            }

        try:
            logs.append(f"Starting case: {case['id']}")
            logs.append(f"Description: {case['description']}")

            # Run the case with timeout
            result = await asyncio.wait_for(
                self._execute_case(case, deployment),
                timeout=timeout,
            )
            logs.extend(result.get("logs", []))
            return {
                "passed": result.get("passed", False),
                "case_id": case_id,
                "logs": logs,
                "error": result.get("error", ""),
            }
        except asyncio.TimeoutError:
            logs.append(f"Case timed out after {timeout}s")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": f"Timeout after {timeout}s",
            }
        except Exception as e:
            logs.append(f"Case error: {e}")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": str(e),
            }

    def _find_case(self, case_id: str) -> dict | None:
        for case in self._registry:
            if case["id"] == case_id:
                return case
        return None

    async def _execute_case(self, case: dict,
                            deployment: dict) -> dict[str, Any]:
        """Execute a fixed case using only the Agent Core HTTP API.

        No subprocess, no shell, no SSH. Only uses the existing
        Agent Core HTTP client for driver status and MCP health checks.

        deployment is a dict with keys (not a Deployment model):
        - component_id, target, variant, driver_path, node_id, node_host,
          image_ref

        Uses dict keys only - no .image_family, .image_digest attribute access.
        """
        case_id = case["id"]
        logs: list[str] = []
        target = case.get("target", "")

        required_keys = {
            "component_id", "target", "variant", "driver_path", "node_id",
            "node_host",
            "image_ref", "machine_alias", "_core", "_driver_id",
        }
        missing = [k for k in required_keys if k not in deployment]
        if missing:
            logs.append(f"FAIL: deployment missing required keys: {', '.join(sorted(missing))}")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": f"deployment missing required keys: {', '.join(sorted(missing))}",
            }

        core = deployment.get("_core")
        driver_id = str(deployment.get("_driver_id") or "")
        if core is None:
            logs.append("FAIL: No Agent Core client available")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "No Agent Core client available",
            }
        if not driver_id:
            logs.append("FAIL: No runtime driver identifier available")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "No runtime driver identifier available",
            }
        if not str(deployment.get("target", "") or ""):
            logs.append("FAIL: deployment target is empty")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment target is empty",
            }
        if not str(deployment.get("node_id", "") or ""):
            logs.append("FAIL: deployment node_id is empty")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment node_id is empty",
            }
        if not str(deployment.get("image_ref", "") or ""):
            logs.append("FAIL: deployment image_ref is empty")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment image_ref is empty",
            }
        if not str(deployment.get("component_id", "") or ""):
            logs.append("FAIL: deployment component_id is empty")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment component_id is empty",
            }
        if not str(deployment.get("node_host", "") or ""):
            logs.append("FAIL: deployment node_host is empty")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment node_host is empty",
            }
        if not str(deployment.get("machine_alias", "") or ""):
            logs.append("FAIL: deployment machine_alias is empty")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment machine_alias is empty",
            }
        if str(deployment.get("target", "") or "") != target:
            logs.append(
                f"FAIL: deployment target {deployment.get('target', '')!r} does not match case target {target!r}"
            )
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": "deployment target mismatch",
            }

        try:
            image_ref = str(deployment.get("image_ref", "") or "")

            try:
                driver_status = await core.driver_status(driver_id)
            except Exception as e:
                logs.append(f"FAIL: driver_status call failed: {e}")
                return {
                    "passed": False,
                    "case_id": case_id,
                    "logs": logs,
                    "error": f"driver_status failed: {e}",
                }

            running_image = str(driver_status.get("running_image", "") or "")
            if not running_image:
                logs.append("FAIL: Running image is empty")
                return {
                    "passed": False,
                    "case_id": case_id,
                    "logs": logs,
                    "error": "Running image is empty",
                }
            if running_image != image_ref:
                logs.append(
                    f"FAIL: Running image {running_image} != expected {image_ref}"
                )
                return {
                    "passed": False,
                    "case_id": case_id,
                    "logs": logs,
                    "error": f"Running image mismatch: {running_image}",
                }
            logs.append("PASS: Expected image is bound")

            if case.get("requires_mcp", False):
                try:
                    drivers = await core.list_drivers()
                except Exception as e:
                    logs.append(f"FAIL: list_drivers failed: {e}")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": f"list_drivers failed: {e}",
                    }

                bound_driver = None
                for d in drivers:
                    if d.get("id", "") == driver_id:
                        bound_driver = d
                        break
                if bound_driver is None:
                    logs.append("FAIL: Bound driver not found in catalog")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": "Bound driver not found in Agent Core catalog",
                    }

                mcp_ref = str(
                    bound_driver.get("mcp_id")
                    or bound_driver.get("mcp_url")
                    or ""
                ).strip()
                if not mcp_ref:
                    logs.append("FAIL: Bound driver has no MCP reference")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": "Bound driver has no MCP reference configured",
                    }

                try:
                    mcps = await core.list_mcp()
                except Exception as e:
                    logs.append(f"FAIL: list_mcp failed: {e}")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": f"list_mcp failed: {e}",
                    }

                matched_mcp = None
                for mcp in mcps:
                    mcp_id = str(mcp.get("id") or "").strip()
                    mcp_url = str(mcp.get("url") or mcp.get("mcp_url") or "").rstrip("/")
                    if mcp_ref == mcp_id or (mcp_url and mcp_ref.rstrip("/") == mcp_url):
                        matched_mcp = mcp
                        break
                if matched_mcp is None:
                    logs.append(f"FAIL: No MCP matches bound driver reference {mcp_ref}")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": "No MCP matches bound driver reference",
                    }

                mcp_id = str(matched_mcp.get("id") or "").strip()
                if not mcp_id:
                    logs.append("FAIL: Matched MCP has no id")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": "Matched MCP has no id",
                    }

                try:
                    ping = await core.mcp_ping(mcp_id)
                except Exception as e:
                    logs.append(f"FAIL: MCP {mcp_id} ping failed: {e}")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": f"MCP {mcp_id} ping failed: {e}",
                    }
                if ping.get("online") is not True:
                    logs.append(f"FAIL: MCP {mcp_id} is offline")
                    return {
                        "passed": False,
                        "case_id": case_id,
                        "logs": logs,
                        "error": f"MCP {mcp_id} is offline",
                    }
                logs.append(f"MCP {mcp_id}: online")

            logs.append(f"PASS: Case {case_id} completed")
            return {
                "passed": True,
                "case_id": case_id,
                "logs": logs,
                "error": "",
            }

        except Exception as e:
            logs.append(f"Case execution error: {e}")
            return {
                "passed": False,
                "case_id": case_id,
                "logs": logs,
                "error": str(e),
            }
