#!/usr/bin/env python3
"""Run a benchmark with conservative macOS unified-memory safety limits."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import subprocess
import sys
import time


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def vm_pages() -> dict[str, int]:
    output = subprocess.check_output(["vm_stat"], text=True)
    result: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r'([^:]+):\s+([0-9]+)\.?$', line)
        if match:
            result[match.group(1)] = int(match.group(2))
    return result


def available_gib() -> float:
    pages = vm_pages()
    reclaimable = sum(
        pages.get(name, 0)
        for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    )
    return reclaimable * PAGE_SIZE / (1024**3)


def process_rows() -> list[tuple[int, int, int, int]]:
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,pgid=,rss="], text=True
    )
    return [tuple(map(int, line.split())) for line in output.splitlines()]


def guarded_pids(root_pid: int) -> set[int]:
    rows = process_rows()
    protected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, pgid, _ in rows:
            if (ppid in protected or pgid == root_pid) and pid not in protected:
                protected.add(pid)
                changed = True
    return protected


def guarded_tree_rss_gib(root_pid: int) -> float:
    try:
        rows = process_rows()
        protected = guarded_pids(root_pid)
        total_kib = sum(rss for pid, _, _, rss in rows if pid in protected)
        return total_kib / (1024**2)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0.0


def stop_tree(pid: int) -> None:
    try:
        protected = guarded_pids(pid)
    except subprocess.CalledProcessError:
        protected = {pid}
    for child_pid in sorted(protected - {pid}, reverse=True):
        try:
            os.kill(child_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for child_pid in sorted(protected - {pid}, reverse=True):
        try:
            os.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-start-gib", type=float, default=42.0)
    parser.add_argument("--min-available-gib", type=float, default=8.0)
    parser.add_argument("--max-rss-gib", type=float, default=38.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--lock-file", default="/tmp/qwen38-memory-guard.lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    lock = open(args.lock_file, "w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("memory_guard: refused: another guarded model run is active", file=sys.stderr)
        return 75
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    start_available = available_gib()
    if start_available < args.min_start_gib:
        print(
            f"memory_guard: refused: {start_available:.1f} GiB available; "
            f"requires {args.min_start_gib:.1f} GiB",
            file=sys.stderr,
        )
        return 75

    child = subprocess.Popen(command, start_new_session=True)
    peak_rss = 0.0
    minimum_available = start_available
    try:
        while child.poll() is None:
            current_rss = guarded_tree_rss_gib(child.pid)
            current_available = available_gib()
            peak_rss = max(peak_rss, current_rss)
            minimum_available = min(minimum_available, current_available)
            if current_rss > args.max_rss_gib or current_available < args.min_available_gib:
                print(
                    f"memory_guard: stopping pid {child.pid}: rss={current_rss:.1f} GiB, "
                    f"available={current_available:.1f} GiB",
                    file=sys.stderr,
                )
                stop_tree(child.pid)
                child.wait()
                return 76
            time.sleep(args.interval)
    except KeyboardInterrupt:
        stop_tree(child.pid)
        child.wait()
        return 130
    print(
        f"memory_guard: peak_rss={peak_rss:.1f} GiB "
        f"minimum_available={minimum_available:.1f} GiB",
        file=sys.stderr,
    )
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
