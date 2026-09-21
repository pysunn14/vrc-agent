"""Run in the Windows bridge interpreter; no model imports or device playback."""
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys


def inspect():
    packages = {}
    for name in ("numpy", "scipy", "websockets", "sounddevice", "windows-capture", "Pillow", "psutil"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.system(), "packages": packages, "audio_devices": [], "audio_error": None,
              "dotnet": shutil.which("dotnet"), "process_audio_helper": None, "helper_error": None}
    if packages["numpy"] and packages["scipy"]:
        try:
            sys.path.insert(0, str(Path.cwd()))
            from vrc_ardy_agent.windows_process_audio import _DEFAULT_HELPER
            result["process_audio_helper"] = {"path": str(_DEFAULT_HELPER), "exists": _DEFAULT_HELPER.is_file()}
        except Exception as exc:
            result["helper_error"] = f"{type(exc).__name__}: {exc}"
    if packages["sounddevice"]:
        try:
            import sounddevice
            result["audio_devices"] = [dict(device, index=index) for index, device in enumerate(sounddevice.query_devices())]
            apis = sounddevice.query_hostapis()
            for device in result["audio_devices"]: device["host_api"] = apis[device["hostapi"]]["name"]
        except Exception as exc:
            result["audio_error"] = f"{type(exc).__name__}: {exc}"
    return result


if __name__ == "__main__": print(json.dumps(inspect()))
