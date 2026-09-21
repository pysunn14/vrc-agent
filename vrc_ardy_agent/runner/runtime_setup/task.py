"""Entry point inside the chosen ARDY interpreter; never connects to VRChat."""
import json
import os
from pathlib import Path
import sys


def main():
    action, plan_file, output_file = sys.argv[1:]
    plan = json.loads(Path(plan_file).read_text())
    if action == 'download':
        from .download import prepare_models
        result = prepare_models(plan)
    elif action == 'verify':
        from .verify import runtime_environment, verify_runtime
        os.environ.update(runtime_environment(plan))
        result = verify_runtime(plan)
    else: raise ValueError('unknown setup task')
    from ..settings import atomic_json
    atomic_json(Path(output_file), result)


if __name__ == '__main__': main()
