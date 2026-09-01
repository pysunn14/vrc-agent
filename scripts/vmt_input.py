#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrc_ardy_agent.vmt_input import (
    encode_vmt_button,
    encode_vmt_joystick,
    encode_vmt_trigger,
    hold_joystick,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send VMT virtual controller input over OSC/UDP.")
    parser.add_argument("--host", required=True, help="Windows host running VMT")
    parser.add_argument("--port", type=int, default=39570, help="VMT OSC port")
    sub = parser.add_subparsers(dest="command", required=True)

    joystick = sub.add_parser("joystick", help="Hold a virtual joystick position, then return to neutral")
    joystick.add_argument("--index", type=int, default=1, help="VMT controller index; left controller is normally 1")
    joystick.add_argument("--x", type=float, default=0.0)
    joystick.add_argument("--y", type=float, default=0.0)
    joystick.add_argument("--duration", type=float, default=1.0)
    joystick.add_argument("--rate", type=float, default=20.0)

    trigger = sub.add_parser("trigger", help="Set trigger value, optionally release after a duration")
    trigger.add_argument("--index", type=int, required=True)
    trigger.add_argument("--trigger-index", type=int, default=0)
    trigger.add_argument("--value", type=float, default=1.0)
    trigger.add_argument("--duration", type=float, default=0.2)

    button = sub.add_parser("button", help="Press a VMT button, then release")
    button.add_argument("--index", type=int, required=True)
    button.add_argument("--button-index", type=int, required=True)
    button.add_argument("--duration", type=float, default=0.2)

    return parser.parse_args()


def send_once(packet: bytes, host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, (host, port))
    finally:
        sock.close()


def main() -> None:
    args = parse_args()
    if args.command == "joystick":
        sent = hold_joystick(
            host=args.host,
            port=args.port,
            index=args.index,
            x=args.x,
            y=args.y,
            duration=args.duration,
            rate_hz=args.rate,
        )
        print(f"joystick VMT_{args.index}: x={args.x:g} y={args.y:g}, sent {sent} active packets; neutral sent")
        return

    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")

    if args.command == "trigger":
        active = encode_vmt_trigger(
            index=args.index,
            trigger_index=args.trigger_index,
            timeoffset=0.0,
            value=args.value,
        )
        neutral = encode_vmt_trigger(
            index=args.index,
            trigger_index=args.trigger_index,
            timeoffset=0.0,
            value=0.0,
        )
    else:
        active = encode_vmt_button(
            index=args.index,
            button_index=args.button_index,
            timeoffset=0.0,
            pressed=True,
        )
        neutral = encode_vmt_button(
            index=args.index,
            button_index=args.button_index,
            timeoffset=0.0,
            pressed=False,
        )

    send_once(active, args.host, args.port)
    if args.duration:
        time.sleep(args.duration)
    send_once(neutral, args.host, args.port)
    print(f"{args.command} input sent and released")


if __name__ == "__main__":
    main()