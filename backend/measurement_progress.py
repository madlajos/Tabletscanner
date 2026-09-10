"""Thread-safe, short-lived progress snapshots for automatic measurement."""

from copy import deepcopy
from threading import Lock
from time import monotonic


_lock = Lock()
_operations = {}
_MAX_OPERATIONS = 64


def start(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
        raise ValueError('request_id must be a non-empty string of at most 128 characters.')
    with _lock:
        if len(_operations) >= _MAX_OPERATIONS:
            oldest = min(_operations, key=lambda key: _operations[key]['updated_at'])
            _operations.pop(oldest, None)
        _operations[request_id] = {
            'request_id': request_id,
            'status': 'running',
            'images': [],
            'warnings': [],
            'active_plan_row_index': None,
            'updated_at': monotonic(),
        }


def set_active_plan_row(request_id: str, row_index: int | None) -> None:
    """Publish the zero-based plan row currently used by the hardware."""
    with _lock:
        operation = _operations.get(request_id)
        if operation is None:
            return
        operation['active_plan_row_index'] = row_index
        operation['updated_at'] = monotonic()


def record_image(request_id: str, image: dict) -> None:
    with _lock:
        operation = _operations.get(request_id)
        if operation is None:
            return
        operation['images'].append(deepcopy(image))
        operation['updated_at'] = monotonic()


def record_warning(request_id: str, warning: dict) -> None:
    with _lock:
        operation = _operations.get(request_id)
        if operation is None:
            return
        operation['warnings'].append(deepcopy(warning))
        operation['updated_at'] = monotonic()


def finish(request_id: str, status: str) -> None:
    with _lock:
        operation = _operations.get(request_id)
        if operation is None:
            return
        operation['status'] = status
        operation['active_plan_row_index'] = None
        operation['updated_at'] = monotonic()


def snapshot(request_id: str):
    with _lock:
        operation = _operations.get(request_id)
        if operation is None:
            return None
        result = deepcopy(operation)
        result.pop('updated_at', None)
        return result


def pending_snapshot(request_id: str) -> dict:
    """Represent a poll that reached Flask before the step POST starts."""
    return {
        'request_id': request_id,
        'status': 'pending',
        'images': [],
        'warnings': [],
        'active_plan_row_index': None,
    }
