#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.vrchat_osc import hold_axis, press_button


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send VRChat OSC input commands.")
    parser.add_argument("--host", required=True, help="Windows host running VRChat")
    parser.add_argument("--port", type=int, default=9000, help="VRChat OSC receive port")
    sub = parser.add_subparsers(dest="command", required=True)

    axis = sub.add_parser("axis", help="Hold a VRChat input axis, then return it to zero")
    axis.add_argument("name", choices=["Vertical", "Horizontal", "LookVertical", "LookHorizontal"])
    axis.add_argument("value", type=float)
    axis.add_argument("--duration", type=float, default=1.0)

    button = sub.add_parser("button", help="Press and release a VRChat input button")
    button.add_argument(
        "name",
        choices=[
            "UseRight",
            "UseLeft",
            "GrabRight",
            "GrabLeft",
            "Jump",
            "QuickMenuToggleRight",
            "QuickMenuToggleLeft",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "axis":
        sent = hold_axis(
            host=args.host,
            port=args.port,
            name=args.name,
            value=args.value,
            duration=args.duration,
        )
        print(f"axis {args.name}={args.value:g}: sent {sent} active packets; neutral sent")
        return

    press_button(host=args.host, port=args.port, name=args.name)
    print(f"button {args.name}: pressed and released")


if __name__ == "__main__":
    main()