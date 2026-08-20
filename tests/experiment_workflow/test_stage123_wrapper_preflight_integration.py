from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_host_facts_and_container_preflight_have_one_boundary() -> None:
    host = (ROOT / "scripts/stage123_host_facts.sh").read_text()
    preflight_path = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_preflight.py"
    preflight = preflight_path.read_text()
    tree = ast.parse(preflight)
    assert "--host-facts" in preflight
    assert "--docker-image-id" not in preflight
    assert "docker" in host and "tmux" in host
    command_heads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != "command" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            command_heads.append(node.args[0].value)
    assert "docker" not in command_heads
    assert "tmux" not in command_heads


def test_host_facts_script_is_atomic_and_host_owned() -> None:
    host = (ROOT / "scripts/stage123_host_facts.sh").read_text()
    assert "mv -f" in host
    assert "host_owned" in host
    assert '"docker", "image", "inspect"' in host
