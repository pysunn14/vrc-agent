from types import SimpleNamespace
from scripts.run_windows_companion import _build_sensor_session


def test_calibration_only_attaches_identity_without_audio_or_capture():
    session = _build_sensor_session(SimpleNamespace(calibration_only=True),
        window=SimpleNamespace(process_id=42), epoch=3, input_hub=None)
    session.start()
    assert session.snapshot().running
    assert session.snapshot().components['mode'] == 'calibration_only'
    session.stop()
    assert not session.snapshot().running
