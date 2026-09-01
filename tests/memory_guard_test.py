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


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "devtools" / "memory_guard.py"


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class MemoryGuardSignalTest(unittest.TestCase):
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
