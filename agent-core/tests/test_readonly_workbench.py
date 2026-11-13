"""Sidebar affordance unit test; real browser coverage is the Card E2E tool."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "kind,readonly,protocol,expected",
    [
        ("processor", True, "motus.workbench/1", 1),
        ("actuator", True, "motus.workbench/1", 0),
        ("processor", False, "motus.workbench/1", 0),
        ("processor", True, "unknown", 0),
    ],
)
def test_sidebar_readonly_workbench_does_not_start_project(
    kind, readonly, protocol, expected
):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for this JavaScript unit test")
    source = (Path(__file__).parents[1] / "web/js/sidebar.js").read_text()
    begin = source.index("function _readOnlyWorkbenchButton(")
    end = source.index("\n}\n", begin) + 2
    function = source[begin:end]
    tool = dict(
        type=kind,
        inputSchema={"x-workbench": {"protocol": protocol, "readOnly": readonly}},
    )
    program = (
        """let buttons=[];let requests=0;
const document={createElement:()=>({addEventListener:(event,callback)=>{},})};
const alert=()=>{};const fetch=()=>{requests++;};
"""
        + function
        + '\n_readOnlyWorkbenchButton({appendChild:b=>buttons.push(b)},{id:"test"},'
        + json.dumps(tool)
        + ");\n"
        + """
console.log(JSON.stringify({count:buttons.length,requests}));"""
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", program],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == dict(count=expected, requests=0)
