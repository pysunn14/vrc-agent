"""Execute an explicit sample on the selected model interpreter, with no devices."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vrc_ardy_agent.runner.profile import Profile
from vrc_ardy_agent.runner.diagnostics import provider_probe


if __name__ == "__main__":
    try:
        profile = Profile.load(sys.argv[1])
        payload = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            result = provider_probe(profile, payload["capability"], text=payload.get("text"), audio_file=payload.get("audio_file"))
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
