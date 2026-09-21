"""Headless JSON boundary used by the npm CLI and development agentctl.

Every invocation observes persisted state; terminal sessions never own service
lifetimes. stdout is an event protocol, with logs and heartbeats on stderr.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from filelock import FileLock

from ..providers.catalog import catalog, definition, validate_connection, validate_binding
from .core import Runner
from .diagnostics import current_os, doctor, list_provider_items, provider_probe_on_runtime, runtime_probe
from .launch import bridge_command, runner_config, connection_info
from .profile import Profile, ProfileStore
from .reproducibility import snapshot


def dispatch(action, payload, *, path=None, state_dir=None, emit=lambda *_: None, _locked=False):
    if not isinstance(payload, dict): raise ValueError("payload must be an object")
    store = ProfileStore(path)
    if action in ("profile.commit", "settings", "start", "stop", "snapshot", "providers.remove") and not _locked:
        store.path.parent.mkdir(parents=True, exist_ok=True)
        # Configuration publication and service starts share this lock. Otherwise
        # a process can load a profile while a different client replaces it.
        with FileLock(str(store.path) + ".operation.lock", timeout=20):
            return dispatch(action, payload, path=path, state_dir=state_dir, emit=emit, _locked=True)
    if action == "bootstrap":
        saved = store.read()
        try: draft = json.loads(store.draft_path.read_text(encoding="utf-8"))
        except FileNotFoundError: draft = None
        if draft:
            from .profile_upgrade import upgrade_profile
            draft["profile"] = upgrade_profile(draft["profile"], draft=True)
        return saved | {"protocol": 3, "catalog": catalog(), "draft": draft, "path": str(store.path),
                        "os": current_os(), "python": sys.executable, "cwd": str(Path.cwd())}
    if action.startswith("calibration."):
        from .calibration import calibration_action
        return calibration_action(action, payload, base=store.path.parent)
    if action.startswith("avatars."):
        from .avatar_packages import package_action
        return package_action(action, payload, base=store.path.parent)
    if action in ("connection.test", "connection.observation"):
        from .connection_tests import connection_test
        return connection_test(action, payload, state_dir=state_dir)
    if action == "connection.describe":
        return definition(payload["provider"], validate_connection(payload["provider"], payload["config"]))
    if action == "hosts.discover":
        from .host_discovery import discover_hosts
        return discover_hosts()
    if action in ("connection.auth.prepare", "connection.auth.save"):
        from ..providers.credentials import prepare, save_key
        config = validate_connection(payload["provider"], payload["config"])
        if action.endswith("prepare"): return prepare(payload["provider"], config)
        return save_key(payload["provider"], config, payload["key"])
    if action == "connection.discover":
        from ..providers.discovery import discover_items
        return discover_items(payload["provider"], payload["config"], kind=payload["kind"],
                              capability=payload.get("capability"), cursor=payload.get("cursor"))
    if action == "bridge.discover":
        from .bridge_diagnostics import inspect_bridge_environment
        paths = dispatch("bridge.paths", payload, path=path, state_dir=state_dir)
        return inspect_bridge_environment(paths["python"], paths["project_dir"])
    if action == "bridge.paths":
        from .runtime_setup.paths import absolute_path, discover_python
        project = absolute_path(payload["project_dir"], payload.get("base_dir"))
        if not (project / "scripts" / "run_windows_companion.py").is_file():
            raise ValueError("project directory does not contain scripts/run_windows_companion.py")
        executable = discover_python(absolute_path(payload["python"], payload.get("base_dir")))
        return {"project_dir": str(project), "python": str(executable)}
    if action.startswith("runtime.") and action != "runtime.probe":
        from .runtime_setup.api import runtime_action
        return runtime_action(action, payload, store=store, state_dir=state_dir)
    if action in ("avatar.rig.inspect", "avatar.files.inspect", "avatar.attest", "avatar.inspect", "behaviors.import", "behaviors.attest", "behaviors.inspect"):
        from .avatar_setup import setup_action
        return setup_action(action, payload, base=store.path.parent)
    if action == "catalog": return catalog()
    if action == "connection.validate": return validate_connection(payload["provider"], payload["config"])
    if action == "binding.validate": return validate_binding(payload["provider"], payload["capability"], payload["settings"], config=payload.get("config"))
    if action == "profile.validate": return Profile.parse(payload["profile"], store.path).data
    if action == "profile.commit":
        existing = store.read()["profile"]
        proposed = Profile.parse(payload["profile"], store.path).data
        if existing and {k: v for k, v in existing.items() if k != "language"} != {k: v for k, v in proposed.items() if k != "language"}:
            runner = Runner(runner_config(Profile.parse(existing, store.path)), state_dir)
            owned = [s.id for s in runner.config.services if runner.observe(s.id, check_health=False)["owned"]]
            if owned: raise ValueError(f"stop owned services before changing execution settings: {', '.join(owned)}")
        return store.commit(proposed, expected_revision=payload["revision"])
    if action == "draft.save":
        store.save_draft(payload["profile"], expected_revision=payload["revision"], completed=payload.get("completed", []))
        return {"saved": True, "path": str(store.draft_path)}
    saved = store.read()
    if saved["profile"] is None: raise ValueError("profile is not configured; run vrc-agent onboard")
    profile = Profile.parse(saved["profile"], store.path)
    if action == "behaviors" or action.startswith("providers") or action == "runtime.probe": profile.require_agent()
    if action == "connection.info": return connection_info(profile)
    if action == "behaviors":
        from .avatar_setup import inspect_assets
        result = inspect_assets(profile)
        return {"available": [spec.describe() for spec in result["available"].values()],
                "unavailable": result["unavailable"], "avatar": result["avatar"], "autonomy": profile.data["autonomy"]}
    if action == "settings":
        if "language" not in payload: return {"language": profile.data["language"]}
        profile.data["language"] = payload["language"]
        return store.commit(profile.data, expected_revision=saved["revision"])
    if action == "bridge.command":
        command = bridge_command(profile)
        return {"host": profile.data["game_host"], "command": command,
                "powershell": "& " + " ".join("'" + value.replace("'", "''") + "'" for value in command)}
    if action == "providers":
        return {"connections": profile.data["providers"], "bindings": profile.data["bindings"]}
    if action == "providers.remove":
        instance = payload["instance"]
        if instance not in profile.data["providers"]: raise ValueError("unknown provider instance")
        if any(binding["instance"] == instance for binding in profile.data["bindings"].values()):
            raise ValueError("rebind capabilities before removing this connection")
        del profile.data["providers"][instance]
        dispatch("profile.commit", {"profile": profile.data, "revision": saved["revision"]},
                 path=path, state_dir=state_dir, emit=emit, _locked=True)
        return {"removed": instance}
    if action == "providers.list": return list_provider_items(profile, payload["instance"], payload["kind"], payload.get("cursor"))
    if action in ("providers.check", "providers.observations"):
        from .connection_tests import connection_test
        capability=payload['capability']
        if capability not in ('llm','stt','tts'): raise ValueError('unknown capability')
        connection,binding=profile.resolve(capability)
        args={'provider':connection['provider'],'config':connection['config'],'capability':capability,
              'settings':{k:v for k,v in binding.items() if k!='instance'},
              'runtime':{k:profile.data['runtime'][k] for k in ('python','project_dir')},
              **{k:payload[k] for k in ('text','audio_file') if payload.get(k)}}
        return connection_test('connection.test' if action=='providers.check' else 'connection.observation',args,state_dir=state_dir)
    if action == "providers.test": return provider_probe_on_runtime(profile, payload)
    if action == "runtime.probe": return runtime_probe(profile)
    progress = lambda done, total: emit({"event": "progress", "completed": done, "total": total})
    if action == "doctor": return doctor(profile, offline=payload.get("offline", False), state_dir=state_dir, progress=progress)
    if action == "snapshot":
        return snapshot(profile, saved["revision"], progress=progress)
    runner = Runner(runner_config(profile), state_dir)
    if action == "status":
        return [row | {"local": bool(runner.service(row["id"]).command)} for row in runner.status(progress=progress)]
    if action == "logs":
        if not runner.service(payload["service"]).command: raise ValueError("read remote service logs on its owning host")
        return {"service": payload["service"], "output": runner.logs(payload["service"])}
    if action in ("start", "stop"):
        expected_os = profile.data["hosts"][profile.local_host]["os"]
        if expected_os != current_os(): raise ValueError("this profile belongs to a different execution host OS")
        target = payload["service"]
        if not runner.service(target).command: raise ValueError(f"{target} is remote or externally managed; operate it on its owning host")
        if action == "start" and target == "bridge":
            checks = doctor(profile, offline=True, state_dir=state_dir)
            errors = [row for row in checks if row["result"] == "error" and row["service"] in ("profile", "runner", "bridge")]
            if errors: raise ValueError("bridge preflight failed: " + json.dumps(errors, ensure_ascii=False))
        if action == "start" and target == "agent":
            checks = doctor(profile, offline=True, state_dir=state_dir)
            errors = [row for row in checks if row["result"] == "error" and row["service"] != "bridge"]
            if errors: raise ValueError("agent preflight failed: " + json.dumps(errors, ensure_ascii=False))
            dependencies = {binding["instance"] for binding in profile.data["bindings"].values()}
            for instance in sorted(dependencies):
                deploy = profile.data["providers"][instance]["deployment"]
                if deploy["kind"] != "managed": continue
                status = runner.observe("provider-" + instance)
                if not status["owned"] and status["state"] != "external":
                    raise ValueError(f"start provider-{instance} before starting agent")
                if deploy.get("health_url") and not status["health"]["ok"]:
                    raise ValueError(f"provider-{instance} is not healthy")
        return getattr(runner, action)(target)
    raise ValueError(f"unknown core action: {action}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="VRC AGENT headless core (JSON stdin / NDJSON stdout)")
    parser.add_argument("action")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args(argv)
    output_lock = threading.Lock()
    stopped = threading.Event()
    def emit(value):
        with output_lock: print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    def heartbeat():
        while not stopped.wait(1): emit({"event": "heartbeat", "action": args.action, "at": time.time()})
    thread = threading.Thread(target=heartbeat, daemon=True)
    try:
        raw = sys.stdin.read(2*1024*1024+1)
        if len(raw) > 2*1024*1024: raise ValueError("request too large")
        payload = json.loads(raw) if raw.strip() else {}
        thread.start()
        result = dispatch(args.action, payload, path=args.config, state_dir=args.state_dir, emit=emit)
        emit({"event": "result", "data": result})
        return 0
    except Exception as exc:
        emit({"event": "error", "type": type(exc).__name__, "message": str(exc)})
        return 2
    finally:
        stopped.set()
        if thread.is_alive(): thread.join(timeout=1)


if __name__ == "__main__": raise SystemExit(main())
