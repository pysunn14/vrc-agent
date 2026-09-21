from dataclasses import dataclass
from enum import StrEnum
import json
import unittest

from vrc_ardy_agent.status_serialization import to_jsonable


class _State(StrEnum):
    READY = "ready"


@dataclass(frozen=True)
class _Snapshot:
    state: _State
    payload: bytes


class StatusSerializationTests(unittest.TestCase):
    def test_binary_status_is_summarized_without_exposing_payload(self) -> None:
        value = to_jsonable({"snapshot": _Snapshot(_State.READY, b"secret")})

        self.assertEqual(
            value,
            {
                "snapshot": {
                    "state": "ready",
                    "payload": {"type": "bytes", "size": 6},
                }
            },
        )
        self.assertNotIn("secret", json.dumps(value))


if __name__ == "__main__":
    unittest.main()
