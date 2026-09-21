from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import sys

from .windows_capture import WindowInfo, choose_window, list_windows


_LOG_NAME_PATTERN = re.compile(
    r"^output_log_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.txt$"
)
_AUTHENTICATED_PATTERN = re.compile(
    r"User Authenticated:\s*(?P<display_name>.+?)\s*\((?P<user_id>usr_[^)]+)\)"
)


class AmbiguousVrchatIdentityError(RuntimeError):
    """Raised when an account selector matches multiple live processes."""


@dataclass(frozen=True, slots=True)
class VrchatIdentity:
    display_name: str
    user_id: str


@dataclass(frozen=True, slots=True)
class VrchatTarget:
    window: WindowInfo
    process_started_at: float
    log_path: Path | str
    identity: VrchatIdentity

    @property
    def generation_key(self) -> tuple[int, float]:
        return self.window.process_id, self.process_started_at


@dataclass(frozen=True, slots=True)
class _LogSession:
    started_at: float
    path: Path
    identity: VrchatIdentity


class VrchatIdentityResolver:
    """Resolve a live VRChat window from its authenticated account log."""

    def __init__(
        self,
        *,
        user_name: str | None = None,
        user_id: str | None = None,
        log_dir: str | Path | None = None,
        window_title: str = "VRChat",
        maximum_start_delta_seconds: float = 30.0,
        maximum_log_files: int = 100,
        enumerate_windows: Callable[..., list[WindowInfo]] = list_windows,
        process_started_at: Callable[[int], float] | None = None,
    ) -> None:
        normalized_name = user_name.strip() if user_name is not None else None
        normalized_id = user_id.strip() if user_id is not None else None
        if bool(normalized_name) == bool(normalized_id):
            raise ValueError("exactly one of user_name or user_id is required")
        normalized_title = window_title.strip()
        if not normalized_title:
            raise ValueError("window_title must not be empty")
        if maximum_start_delta_seconds <= 0:
            raise ValueError("maximum_start_delta_seconds must be positive")
        if maximum_log_files <= 0:
            raise ValueError("maximum_log_files must be positive")

        self.user_name = normalized_name
        self.user_id = normalized_id
        self.log_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
        self.window_title = normalized_title
        self.maximum_start_delta_seconds = float(maximum_start_delta_seconds)
        self.maximum_log_files = int(maximum_log_files)
        self._enumerate_windows = enumerate_windows
        self._process_started_at = process_started_at or windows_process_started_at
        self._log_cache: dict[Path, tuple[int, int, _LogSession | None]] = {}

    def resolve(self) -> VrchatTarget | None:
        windows = self._enumerate_windows(title_filter=self.window_title)
        process_windows: dict[int, list[WindowInfo]] = {}
        process_starts: dict[int, float] = {}
        for window in windows:
            process_windows.setdefault(window.process_id, []).append(window)
        for process_id in tuple(process_windows):
            try:
                process_starts[process_id] = self._process_started_at(process_id)
            except (OSError, ProcessLookupError):
                process_windows.pop(process_id, None)

        sessions = self._read_log_sessions()
        matches: list[VrchatTarget] = []
        for process_id, candidates in process_windows.items():
            started_at = process_starts[process_id]
            session = _nearest_session(
                sessions,
                started_at=started_at,
                maximum_delta_seconds=self.maximum_start_delta_seconds,
            )
            if session is None or not self._identity_matches(session.identity):
                continue
            window = choose_window(candidates, process_id=process_id)
            if window is None:
                continue
            matches.append(
                VrchatTarget(
                    window=window,
                    process_started_at=started_at,
                    log_path=session.path,
                    identity=session.identity,
                )
            )

        if len(matches) > 1:
            pids = ", ".join(str(match.window.process_id) for match in matches)
            selector = self.user_id or self.user_name
            raise AmbiguousVrchatIdentityError(
                f"VRChat identity {selector!r} matched multiple live processes: {pids}"
            )
        return matches[0] if matches else None

    def is_current(self, target: VrchatTarget) -> bool:
        try:
            current_started_at = self._process_started_at(target.window.process_id)
        except (OSError, ProcessLookupError):
            return False
        return abs(current_started_at - target.process_started_at) < 0.001

    def diagnostics(self) -> dict[str, object]:
        """Describe live window-to-log matching without exposing account IDs."""
        windows = self._enumerate_windows(title_filter=self.window_title)
        process_windows: dict[int, list[WindowInfo]] = {}
        for window in windows:
            process_windows.setdefault(window.process_id, []).append(window)
        sessions = self._read_log_sessions()
        processes: list[dict[str, object]] = []
        matching_processes = 0
        for process_id, candidates in sorted(process_windows.items()):
            try:
                started_at = self._process_started_at(process_id)
            except (OSError, ProcessLookupError) as exc:
                processes.append(
                    {
                        "process_id": process_id,
                        "window_count": len(candidates),
                        "inspection_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            nearest = min(
                sessions,
                key=lambda session: abs(session.started_at - started_at),
                default=None,
            )
            delta = (
                abs(nearest.started_at - started_at)
                if nearest is not None
                else None
            )
            within_tolerance = (
                delta is not None
                and delta <= self.maximum_start_delta_seconds
            )
            identity_matches = bool(
                nearest is not None
                and within_tolerance
                and self._identity_matches(nearest.identity)
            )
            matching_processes += int(identity_matches)
            selected_window = choose_window(candidates, process_id=process_id)
            processes.append(
                {
                    "process_id": process_id,
                    "window_count": len(candidates),
                    "selected_hwnd": (
                        selected_window.hwnd if selected_window is not None else None
                    ),
                    "process_started_at": started_at,
                    "nearest_log": nearest.path.name if nearest is not None else None,
                    "nearest_display_name": (
                        nearest.identity.display_name if nearest is not None else None
                    ),
                    "start_delta_seconds": delta,
                    "within_start_tolerance": within_tolerance,
                    "identity_matches": identity_matches,
                }
            )
        return {
            "selector": self.user_id or self.user_name,
            "selector_type": "user_id" if self.user_id is not None else "display_name",
            "window_count": len(windows),
            "session_count": len(sessions),
            "matching_processes": matching_processes,
            "processes": processes,
        }

    def _identity_matches(self, identity: VrchatIdentity) -> bool:
        if self.user_id is not None:
            return identity.user_id == self.user_id
        assert self.user_name is not None
        return identity.display_name.casefold() == self.user_name.casefold()

    def _read_log_sessions(self) -> tuple[_LogSession, ...]:
        if not self.log_dir.is_dir():
            raise FileNotFoundError(f"VRChat log directory does not exist: {self.log_dir}")
        paths = sorted(
            self.log_dir.glob("output_log_*.txt"),
            key=lambda path: path.name,
            reverse=True,
        )[: self.maximum_log_files]
        current_paths = set(paths)
        for stale in set(self._log_cache) - current_paths:
            self._log_cache.pop(stale, None)

        sessions: list[_LogSession] = []
        for path in paths:
            stat = path.stat()
            cached = self._log_cache.get(path)
            cache_key = (stat.st_mtime_ns, stat.st_size)
            if cached is not None and cached[:2] == cache_key:
                session = cached[2]
            else:
                session = _parse_log_session(path)
                self._log_cache[path] = (*cache_key, session)
            if session is not None:
                sessions.append(session)
        return tuple(sessions)


def _nearest_session(
    sessions: tuple[_LogSession, ...],
    *,
    started_at: float,
    maximum_delta_seconds: float,
) -> _LogSession | None:
    ranked = sorted(
        (
            (abs(session.started_at - started_at), session)
            for session in sessions
            if abs(session.started_at - started_at) <= maximum_delta_seconds
        ),
        key=lambda item: (item[0], item[1].path.name),
    )
    if not ranked:
        return None
    if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 0.001:
        raise AmbiguousVrchatIdentityError(
            f"process start time {started_at} matched two VRChat logs equally"
        )
    return ranked[0][1]


def _parse_log_session(path: Path) -> _LogSession | None:
    name_match = _LOG_NAME_PATTERN.match(path.name)
    if name_match is None:
        return None
    started_at = datetime.strptime(
        name_match.group("stamp"),
        "%Y-%m-%d_%H-%M-%S",
    ).timestamp()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        prefix = stream.read(512 * 1024)
    matches = tuple(_AUTHENTICATED_PATTERN.finditer(prefix))
    if not matches:
        return None
    authenticated = matches[-1]
    return _LogSession(
        started_at=started_at,
        path=path,
        identity=VrchatIdentity(
            display_name=authenticated.group("display_name").strip(),
            user_id=authenticated.group("user_id").strip(),
        ),
    )


def _default_log_dir() -> Path:
    user_profile = os.environ.get("USERPROFILE")
    if not user_profile:
        raise RuntimeError("USERPROFILE is required to locate VRChat logs")
    return Path(user_profile) / "AppData" / "LocalLow" / "VRChat" / "VRChat"


def windows_process_started_at(process_id: int) -> float:
    if sys.platform != "win32":
        raise RuntimeError("Windows process inspection is only available on Windows")
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        raise ValueError("process_id must be a positive integer")

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            raise ProcessLookupError(process_id)
        raise ctypes.WinError(error)
    try:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        filetime = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return filetime / 10_000_000.0 - 11_644_473_600.0
    finally:
        kernel32.CloseHandle(handle)
