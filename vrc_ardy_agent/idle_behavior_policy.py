"""Quiet-time scheduling, independent of action names and body playback."""
from .asset_contract import number


class IdleBehaviorPolicy:
    """Called under the interaction activity lock; no independent action writer."""
    def __init__(self, config, available):
        self.enabled = config['enabled']
        self.interval = number(config['interval_seconds'], 1, 86400)
        self.pool = tuple(key for key in config['behaviors'] if key in available)
        self.unavailable = tuple(key for key in config['behaviors'] if key not in available)
        self._idle_since = None
        self._cursor = 0
        self._state = 'waiting' if self.enabled and self.pool else 'unavailable' if self.enabled else 'disabled'
        self._last_action_id = None
        self._last_error = None
        self._heartbeat = 0.

    def interrupt(self):
        self._idle_since = None
        if self.enabled and self.pool and not self._last_error: self._state = 'blocked'

    def tick(self, *, now, eligible, submit):
        self._heartbeat = now
        if not self.enabled or not self.pool or self._last_error: return
        if not eligible:
            self._idle_since = None
            self._state = 'blocked'
            return
        if self._idle_since is None: self._idle_since = now
        self._state = 'waiting'
        if now - self._idle_since < self.interval: return
        try:
            action_id = submit(self.pool[self._cursor])
        except Exception as exc:
            self._state = 'failed'
            self._last_error = f'{type(exc).__name__}: {exc}'
            return
        self._idle_since = None
        if action_id is None:
            self._state = 'blocked'
            return
        self._last_action_id = action_id
        self._cursor = (self._cursor + 1) % len(self.pool)
        self._state = 'submitted'

    def snapshot(self):
        return {'enabled': self.enabled, 'state': self._state, 'available': list(self.pool),
                'unavailable': list(self.unavailable), 'interval_seconds': self.interval,
                'idle_since_monotonic': self._idle_since, 'last_action_id': self._last_action_id,
                'last_error': self._last_error, 'heartbeat_monotonic': self._heartbeat}
