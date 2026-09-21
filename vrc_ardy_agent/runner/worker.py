"""Supervise one explicitly started process independently of the terminal UI."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

import psutil

from .records import read_record, update_record


def terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=3)
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        # Descendants may survive the parent, but belong to the session we own.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=3)
        for child in alive:
            child.kill()
        process.wait(timeout=3)


def copy_output(stream, path: Path) -> None:
    limit = 2 * 1024 * 1024
    output = path.open("ab")
    try:
        while chunk := stream.read1(8192):
            if output.tell() + len(chunk) > limit:
                output.close()
                path.replace(path.with_suffix(".log.1"))
                output = path.open("ab")
            output.write(chunk)
            output.flush()
    finally:
        output.close()


def supervise(path: Path, ticket: str) -> None:
    state = read_record(path)
    if state.get("ticket") != ticket:
        raise RuntimeError("stale process launch ticket")
    process = None
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    me = psutil.Process()
    update_record(path, ticket, supervisor_pid=me.pid, supervisor_created=me.create_time(), heartbeat_at=time.time())
    try:
        spec = state["service"]
        environment = os.environ | spec["env"]
        credential_profile = environment.pop("VRC_AGENT_CREDENTIAL_PROFILE", None)
        if credential_profile:
            from .profile import Profile
            from ..providers.credentials import runtime_environment
            environment = runtime_environment(Profile.load(credential_profile), environment)
        process = subprocess.Popen(
            spec["command"], cwd=spec["cwd"], env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        child = psutil.Process(process.pid)
        update_record(path, ticket, pid=child.pid, created=child.create_time(), phase="running", heartbeat_at=time.time())
        reader_errors = []

        def read_output():
            try:
                copy_output(process.stdout, path.with_suffix(".log"))
            except Exception as exc:
                reader_errors.append(str(exc))

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_heartbeat = 0.
        requested = False
        while process.poll() is None:
            requested = stop_event.is_set() or read_record(path).get("stop_requested", False)
            if reader_errors:
                raise RuntimeError(f"output capture failed: {reader_errors[0]}")
            if requested:
                update_record(path, ticket, phase="stopping")
                terminate_tree(process)
                break
            if time.monotonic() - last_heartbeat >= 1:
                update_record(path, ticket, heartbeat_at=time.time())
                last_heartbeat = time.monotonic()
            stop_event.wait(.1)
        reader.join(timeout=1)
        code = process.wait()
        if reader_errors:
            raise RuntimeError(f"output capture failed: {reader_errors[0]}")
        update_record(path, ticket, phase="stopped" if requested or code == 0 else "failed",
                      exit_code=code, finished_at=time.time(), heartbeat_at=time.time())
    except Exception as exc:
        if process is not None:
            terminate_tree(process)
        update_record(path, ticket, phase="failed", error=f"{type(exc).__name__}: {exc}",
                      exit_code=process.returncode if process is not None else None, finished_at=time.time())
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--ticket", required=True)
    args = parser.parse_args()
    supervise(args.record, args.ticket)
