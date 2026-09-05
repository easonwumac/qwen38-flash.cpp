#!/usr/bin/env python3
"""Integration checks for memory_guard child-process cleanup."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import memory_guard  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "devtools" / "memory_guard.py"


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class MemoryGuardSignalTest(unittest.TestCase):
    def test_main_stops_live_child_on_measurement_failure(self) -> None:
        child = mock.Mock(pid=123, returncode=-signal.SIGTERM)
        child.poll.side_effect = [None, None]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            sys,
            "argv",
            [
                str(GUARD),
                "--min-start-gib", "0",
                "--lock-file", str(Path(directory) / "guard.lock"),
                "--", "dummy",
            ],
        ), mock.patch.object(memory_guard, "available_gib", return_value=100), \
             mock.patch.object(memory_guard.subprocess, "Popen", return_value=child), \
             mock.patch.object(memory_guard.fcntl, "flock"), \
             mock.patch("builtins.open", mock.mock_open()), \
             mock.patch.object(
                 memory_guard,
                 "guarded_tree_rss_gib",
                 side_effect=memory_guard.MeasurementError("failed"),
             ), mock.patch.object(memory_guard, "stop_tree") as stop_tree:
            self.assertEqual(memory_guard.main(), memory_guard.MEASUREMENT_FAILURE_EXIT)
        stop_tree.assert_called_once_with(child.pid)
        child.wait.assert_called_once_with()

    def test_main_tolerates_measurement_failure_after_child_exit(self) -> None:
        child = mock.Mock(pid=123, returncode=0)
        child.poll.side_effect = [None, 0]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            sys,
            "argv",
            [
                str(GUARD),
                "--min-start-gib", "0",
                "--lock-file", str(Path(directory) / "guard.lock"),
                "--", "dummy",
            ],
        ), mock.patch.object(memory_guard, "available_gib", return_value=100), \
             mock.patch.object(memory_guard.subprocess, "Popen", return_value=child), \
             mock.patch.object(memory_guard.fcntl, "flock"), \
             mock.patch("builtins.open", mock.mock_open()), \
             mock.patch.object(
                 memory_guard,
                 "guarded_tree_rss_gib",
                 side_effect=memory_guard.MeasurementError("exited"),
             ), mock.patch.object(memory_guard, "stop_tree") as stop_tree:
            self.assertEqual(memory_guard.main(), 0)
        stop_tree.assert_not_called()
        child.wait.assert_not_called()

    def test_available_excludes_overlapping_vm_counters(self) -> None:
        pages = {
            "Pages free": 10,
            "Pages inactive": 20,
            "Pages speculative": 30,
            "Pages purgeable": 40,
        }
        with mock.patch.object(memory_guard, "vm_pages", return_value=pages), \
             mock.patch.object(memory_guard, "PAGE_SIZE", 1024**3):
            self.assertEqual(memory_guard.available_gib(), 60)

    def test_rss_enumeration_failure_is_not_zero(self) -> None:
        with mock.patch.object(
            memory_guard,
            "process_rows",
            side_effect=subprocess.CalledProcessError(1, ["ps"]),
        ):
            with self.assertRaises(memory_guard.MeasurementError):
                memory_guard.guarded_tree_rss_gib(os.getpid())

    def test_live_footprint_failure_is_not_zero(self) -> None:
        with mock.patch.object(memory_guard, "guarded_pids", return_value={os.getpid()}), \
             mock.patch.object(
                 memory_guard, "physical_footprint_bytes", side_effect=OSError("failed")
             ):
            with self.assertRaises(memory_guard.MeasurementError):
                memory_guard.guarded_tree_footprint_gib(os.getpid())

    def test_exited_pid_footprint_race_is_tolerated(self) -> None:
        missing_pid = 2**30
        with mock.patch.object(memory_guard, "guarded_pids", return_value={missing_pid}), \
             mock.patch.object(
                 memory_guard,
                 "physical_footprint_bytes",
                 side_effect=ProcessLookupError(),
             ):
            self.assertEqual(memory_guard.guarded_tree_footprint_gib(missing_pid), 0)

    def test_stop_tree_falls_back_when_process_enumeration_fails(self) -> None:
        with mock.patch.object(
            memory_guard, "guarded_pids", side_effect=ValueError("invalid ps output")
        ), mock.patch.object(memory_guard.os, "killpg") as killpg, mock.patch.object(
            memory_guard.time, "sleep"
        ):
            memory_guard.stop_tree(123)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(123, signal.SIGTERM), mock.call(123, signal.SIGKILL)],
        )

    def test_physical_footprint_includes_current_process(self) -> None:
        self.assertGreater(memory_guard.physical_footprint_bytes(os.getpid()), 0)

    def test_physical_footprint_limit_stops_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(GUARD),
                    "--min-start-gib",
                    "0",
                    "--min-available-gib",
                    "0",
                    "--max-rss-gib",
                    "1024",
                    "--max-footprint-gib",
                    "0.001",
                    "--interval",
                    "0.02",
                    "--lock-file",
                    str(Path(directory) / "guard.lock"),
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 76)
            self.assertIn("footprint=", result.stderr)

    def test_control_signals_reap_guarded_child(self) -> None:
        signal_sequences = (
            (signal.SIGINT,),
            (signal.SIGTERM,),
            (signal.SIGHUP,),
            (signal.SIGINT, signal.SIGTERM),
        )
        for signals in signal_sequences:
            with self.subTest(signals=signals), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                pid_file = temporary / "child.pid"
                child_code = (
                    "import os, pathlib, sys, time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(60)"
                )
                guard = subprocess.Popen(
                    [
                        sys.executable,
                        str(GUARD),
                        "--min-start-gib",
                        "0",
                        "--min-available-gib",
                        "0",
                        "--max-rss-gib",
                        "1024",
                        "--max-footprint-gib",
                        "1024",
                        "--interval",
                        "0.05",
                        "--lock-file",
                        str(temporary / "guard.lock"),
                        "--",
                        sys.executable,
                        "-c",
                        child_code,
                        str(pid_file),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.addCleanup(
                    lambda process=guard: process.poll() is None and process.kill()
                )
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "guarded child did not start")
                child_pid = int(pid_file.read_text())

                for signum in signals:
                    try:
                        os.kill(guard.pid, signum)
                    except ProcessLookupError:
                        break
                self.assertIn(
                    guard.wait(timeout=5),
                    {128 + signum for signum in signals},
                )
                deadline = time.monotonic() + 2
                while process_exists(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(process_exists(child_pid))


if __name__ == "__main__":
    unittest.main()
