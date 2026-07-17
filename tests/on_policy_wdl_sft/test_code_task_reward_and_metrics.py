import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import time
import unittest
import uuid
import zlib
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import _restore_jsonable
from recipe.on_policy_wdl_sft.code_task.code_extraction import extract_code
from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import compute_score_code_official_aligned
from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import score_livecodebench_official
from recipe.on_policy_wdl_sft.code_task.prepare_deepcoder_preview_dataset import convert_row
from verl.experimental.agent_loop.agent_loop import _default_reward_extra_value
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.trainer.ppo.metric_utils import process_validation_metrics


def wrap_code(code: str) -> str:
    return f"<answer>\n```python\n{code}\n```\n</answer>"


class TestCodeTaskRewardAndMetrics(unittest.TestCase):
    def test_livecodebench_sqlite_index_reads_one_question(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lcb.sqlite")
            payload = {"inputs": ["1\n"], "outputs": ["1\n"]}
            with sqlite3.connect(path) as con:
                con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                con.execute(
                    "CREATE TABLE input_output (question_id TEXT PRIMARY KEY, payload_zlib BLOB NOT NULL) WITHOUT ROWID"
                )
                con.execute("INSERT INTO metadata VALUES ('release_version', 'release_v5')")
                con.execute(
                    "INSERT INTO input_output VALUES (?, ?)",
                    ("q1", sqlite3.Binary(zlib.compress(json.dumps(payload).encode()))),
                )
            with patch.dict(os.environ, {"LCB_INPUT_OUTPUT_INDEX": path}):
                from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import (
                    _resolve_livecodebench_input_output,
                )

                self.assertEqual(
                    _resolve_livecodebench_input_output(
                        {"question_id": "q1", "release_version": "release_v5"}
                    ),
                    payload,
                )

    def test_livecodebench_official_timeout_kills_process_group(self):
        class HangingProcess:
            returncode = None
            pid = os.getpid()

            def communicate(self, timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired("lcb", timeout)
                self.returncode = -signal.SIGKILL
                return "", ""

        proc = HangingProcess()
        with patch("recipe.on_policy_wdl_sft.code_task.official_aligned_reward.subprocess.Popen", return_value=proc), patch(
            "recipe.on_policy_wdl_sft.code_task.official_aligned_reward.os.killpg"
        ) as killpg:
            result = score_livecodebench_official("print(1)", {"input_output": {"inputs": [], "outputs": []}})

        self.assertEqual(result["code_reward_status"], "timeout")
        killpg.assert_called_once_with(proc.pid, signal.SIGKILL)

    def test_strict_extraction_rejects_full_text_fallback(self):
        result = extract_code("```python\ndef f():\n    return 1\n```")
        self.assertFalse(result.ok)
        self.assertEqual(result.source, "strict:no_answer")

        relaxed = extract_code("```python\ndef f():\n    return 1\n```", strict_answer=False)
        self.assertTrue(relaxed.ok)
        self.assertEqual(relaxed.source, "full:fenced_python")

    def test_kodcode_firejail_hides_test_solution_file(self):
        if not os.path.exists("/usr/bin/firejail"):
            self.skipTest("firejail is not installed")

        gt = {
            "verification_method": "kodcode_exec",
            "test": "from solution import *\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n",
        }

        ok = compute_score_code_official_aligned(
            "kodcode_light_rl_10k",
            wrap_code("def add(a, b):\n    return a + b"),
            gt,
        )
        self.assertEqual(ok["code_reward_status"], "pass")
        self.assertEqual(ok["code_reward_sandbox"], "firejail")

        leak = compute_score_code_official_aligned(
            "kodcode_light_rl_10k",
            wrap_code(
                "def add(a, b):\n"
                "    import os\n"
                "    return 5 if os.path.exists('test_solution.py') else 0"
            ),
            gt,
        )
        self.assertEqual(leak["code_reward_status"], "wrong_answer")
        self.assertEqual(leak["score"], -1.0)
        self.assertEqual(leak["acc"], 0.0)

    def test_code_core_acc_exposes_distinct_mean_pass_and_std_at_k(self):
        result = process_validation_metrics(
            ["HumanEval+", "HumanEval+", "HumanEval+", "math"],
            ["p1", "p1", "p1", "m1"],
            {"acc": [1.0, 0.0, 0.0, 0.5]},
            seed=1,
        )

        self.assertAlmostEqual(result["HumanEval+"]["acc"]["mean@3"], 1.0 / 3.0)
        self.assertEqual(result["HumanEval+"]["acc"]["pass@3"], 1.0)
        self.assertAlmostEqual(result["HumanEval+"]["acc"]["std@3"], 2**0.5 / 3.0)
        self.assertIn("mean@1", result["math"]["acc"])

    def test_official_json_restore_preserves_evalplus_tuple_inputs(self):
        restored = _restore_jsonable(
            {
                "base_input": [
                    [
                        {"__tuple__": [4, 5, 6]},
                        {"MSAM": 1, "is": 2, "best": 3},
                    ]
                ],
                "z": {"__complex__": [1.0, -2.0]},
            }
        )

        self.assertIsInstance(restored["base_input"][0][0], tuple)
        self.assertEqual(restored["base_input"][0][0], (4, 5, 6))
        self.assertEqual(restored["z"], complex(1.0, -2.0))

    def test_deepcoder_stdin_exec_uses_negative_labels_and_cleans_orphans(self):
        os.environ["CODE_REWARD_STDIN_CASE_TIMEOUT"] = "1"
        gt = {"verification_method": "stdin_stdout_exec", "inputs": [""], "outputs": ["OK"]}

        wrong = compute_score_code_official_aligned("deepcoder_preview_train", wrap_code("print('BAD')"), gt)
        self.assertEqual(wrong["acc"], 0.0)
        self.assertEqual(wrong["score"], -1.0)

        marker = f"codex_deepcoder_orphan_{uuid.uuid4().hex}"
        child = f"import os, time; os.setsid(); open('/data-1/tmp/{marker}', 'w').write('alive'); time.sleep(120)"
        code = (
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "print('OK')\n"
        )
        result = compute_score_code_official_aligned("deepcoder_preview_train", wrap_code(code), gt)
        self.assertEqual(result["score"], -1.0)
        time.sleep(0.2)
        proc = subprocess.run(
            ["ps", "-eo", "pid,ppid,pgid,sid,cmd"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        leaked = [line for line in proc.stdout.splitlines() if marker in line and "python" in line]
        self.assertEqual(leaked, [])

    def test_deepcoder_stdin_exec_feeds_non_empty_stdin(self):
        os.environ["CODE_REWARD_STDIN_CASE_TIMEOUT"] = "1"
        gt = {
            "verification_method": "stdin_stdout_exec",
            "tests": [
                {"input": "2 3\n", "output": "5\n"},
                {"input": "10 -4\n", "output": "6\n"},
            ],
        }

        result = compute_score_code_official_aligned(
            "deepcoder_preview_train",
            wrap_code("a, b = map(int, input().split())\nprint(a + b)"),
            gt,
        )

        self.assertEqual(result["code_reward_status"], "pass")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["code_reward_num_passed"], 2)
        self.assertEqual(result["code_reward_num_tests"], 2)

    def test_dapo_reward_manager_timeout_returns_negative_code_label(self):
        manager = object.__new__(DAPORewardManager)
        manager.timeout = 0.01
        manager.is_async_reward_score = False
        result = manager._timeout_result("deepcoder_preview_train", "print('stuck')")

        self.assertEqual(result["reward_score"], -1.0)
        self.assertEqual(result["reward_extra_info"]["score"], -1.0)
        self.assertEqual(result["reward_extra_info"]["acc"], 0.0)
        self.assertEqual(result["reward_extra_info"]["code_reward_status"], "timeout")
        self.assertEqual(result["reward_extra_info"]["code_reward_runtime_error"], 0)
        self.assertEqual(result["reward_extra_info"]["code_reward_timeout"], 1)
        self.assertEqual(result["reward_extra_info"]["pred"], "print('stuck')")
        self.assertEqual(result["reward_extra_info"]["verification_method"], "reward_manager_fallback")
        self.assertFalse(result["reward_extra_info"]["official_aligned"])
        self.assertIn("code_reward_sandbox", result["reward_extra_info"])

    def test_dapo_timeout_helper_remains_latency_agnostic(self):
        manager = object.__new__(DAPORewardManager)
        manager.timeout = 0.01
        manager.is_async_reward_score = False
        result = manager._timeout_result("code", "stuck")
        self.assertNotIn("code_reward_latency_seconds", result["reward_extra_info"])

    def test_reward_extra_union_defaults_keep_agent_loop_schema_stable(self):
        infos = [
            {"score": 1.0, "acc": 1.0, "pred": "ok", "verification_method": "evalplus"},
            {"score": -1.0, "acc": 0.0, "code_reward_status": "timeout"},
        ]
        keys = sorted({key for info in infos for key in info.keys()})
        columns = {key: [info.get(key, _default_reward_extra_value(key)) for info in infos] for key in keys}

        self.assertEqual(columns["pred"], ["ok", ""])
        self.assertEqual(columns["verification_method"], ["evalplus", ""])
        self.assertEqual(columns["code_reward_status"], ["missing", "timeout"])

    def test_deepcoder_conversion_keeps_only_stdin_stdout_contract(self):
        function_call_row = {
            "problem": "Return x + 1.",
            "tests": {"fn_name": "add_one", "inputs": [[1]], "outputs": [[2]]},
            "solutions": ["def add_one(x):\n    return x + 1"],
        }
        self.assertIsNone(convert_row(function_call_row, "taco", "train", 0))

        functional_row = {
            "problem": "Return x + 1.",
            "tests": [{"testtype": "functional", "input": "[1]", "output": "2"}],
            "solutions": ["def add_one(x):\n    return x + 1"],
        }
        self.assertIsNone(convert_row(functional_row, "lcbv5", "train", 1))

        stdin_row = {
            "problem": "Read two ints and print their sum.",
            "tests": [{"type": "stdin_stdout", "input": "2 3\n", "output": "5\n"}],
            "solutions": ["```python\na, b = map(int, input().split())\nprint(a + b)\n```"],
        }
        converted = convert_row(stdin_row, "primeintellect", "train", 2)

        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted["reward_model"]["style"], "rule")
        self.assertIn("stdin", converted["prompt"][1]["content"])
        self.assertIn("stdout", converted["prompt"][1]["content"])
        self.assertIn("a, b = map", converted["extra_info"]["reference_answer"])
        self.assertNotIn("```", converted["extra_info"]["reference_answer"])


if __name__ == "__main__":
    unittest.main()
