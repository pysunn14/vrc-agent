from __future__ import annotations

import threading

from .action_contracts import ControlResource, ResourceLease


class ResourceBusyError(RuntimeError):
    pass


class ResourceOwnershipError(RuntimeError):
    pass


class ControlResourceManager:
    """Issues monotonic fencing tokens and enforces one owner per resource."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens = {resource: 0 for resource in ControlResource}
        self._owners: dict[ControlResource, str | None] = {
            resource: None for resource in ControlResource
        }

    def acquire(self, resource: ControlResource, action_id: str) -> ResourceLease:
        if not action_id:
            raise ValueError("action_id must not be empty")
        with self._lock:
            owner = self._owners[resource]
            if owner is not None:
                raise ResourceBusyError(f"{resource.value} is owned by {owner}")
            self._tokens[resource] += 1
            self._owners[resource] = action_id
            return ResourceLease(resource, self._tokens[resource], action_id)

    def invalidate(
        self,
        resource: ControlResource,
        *,
        expected_action_id: str | None = None,
    ) -> ResourceLease:
        with self._lock:
            owner = self._owners[resource]
            if expected_action_id is not None and owner != expected_action_id:
                raise ResourceOwnershipError(
                    f"{resource.value} owner is {owner!r}, expected {expected_action_id!r}"
                )
            if owner is not None:
                self._tokens[resource] += 1
                self._owners[resource] = None
            return ResourceLease(resource, self._tokens[resource], None)

    def current(self, resource: ControlResource) -> ResourceLease:
        with self._lock:
            return ResourceLease(
                resource,
                self._tokens[resource],
                self._owners[resource],
            )

    def is_current(self, lease: ResourceLease) -> bool:
        with self._lock:
            return (
                self._tokens[lease.resource] == lease.token
                and self._owners[lease.resource] == lease.action_id
                and lease.action_id is not None
            )
