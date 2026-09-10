"""State and naming helpers for the manual blue/green/red capture series."""

from dataclasses import dataclass
import os
import re
import threading
import copy


@dataclass(frozen=True)
class FilterCaptureTarget:
    """One configured filter required by the BGR capture button."""

    name: str
    suffix: str
    position: int


REQUIRED_FILTERS = (("Piros", "r"), ("Zöld", "g"), ("Kék", "b"))


class FilterCaptureSeriesCoordinator:
    """Allow one series at a time and expose cooperative cancellation."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._running = False
        self._autofocus_in_progress = False
        self._saved_images = []
        self._camera_params = None
        self._capture_id = None

    def begin(self, capture_id=None) -> bool:
        with self._state_lock:
            if self._running:
                return False
            self._cancel_event.clear()
            self._autofocus_in_progress = False
            self._running = True
            self._saved_images = []
            self._camera_params = None
            self._capture_id = capture_id
            return True

    def record_image(self, image):
        with self._state_lock:
            self._saved_images.append(copy.deepcopy(image))

    def record_camera_params(self, camera_params):
        with self._state_lock:
            if self._running:
                self._camera_params = copy.deepcopy(camera_params)

    def status(self):
        with self._state_lock:
            return {'running': self._running, 'capture_id': self._capture_id,
                    'saved_images': copy.deepcopy(self._saved_images),
                    'camera_params': copy.deepcopy(self._camera_params)}

    def request_cancel(self) -> bool:
        with self._state_lock:
            if not self._running:
                return False
            self._cancel_event.set()
            return True

    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

    def wait_for_cancellation(self, timeout_seconds: float) -> bool:
        return self._cancel_event.wait(timeout_seconds)

    def set_autofocus_in_progress(self, in_progress: bool) -> None:
        with self._state_lock:
            self._autofocus_in_progress = bool(in_progress) and self._running

    def autofocus_in_progress(self) -> bool:
        with self._state_lock:
            return self._running and self._autofocus_in_progress

    def finish(self) -> None:
        with self._state_lock:
            self._running = False
            self._autofocus_in_progress = False
            self._cancel_event.clear()


def resolve_filter_targets(filter_settings: dict, mode='rgb', channel='vis') -> list[FilterCaptureTarget]:
    """Resolve RGB, optionally followed by the UV filter matching the start lamp."""
    if mode not in ('rgb', 'uv_rgb'):
        raise ValueError('Capture mode must be rgb or uv_rgb.')
    if channel not in ('vis', 'uv255', 'uv310', 'uv365'):
        raise ValueError('Select an active illumination channel before capture.')
    definitions = filter_settings.get("filters", [])
    slots = filter_settings.get("slots", [])
    if not isinstance(definitions, list) or not isinstance(slots, list):
        raise ValueError("Filter settings do not contain a valid filter list and slot list.")

    ids_by_name: dict[str, str] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        filter_id = definition.get("id")
        name = definition.get("name")
        if isinstance(filter_id, str) and isinstance(name, str):
            ids_by_name[re.sub(r'[\s_-]+', '', name).casefold()] = filter_id

    targets: list[FilterCaptureTarget] = []
    required = list(REQUIRED_FILTERS)
    if mode == 'uv_rgb' and channel != 'vis':
        required.append(('255nm' if channel in ('uv255', 'uv310') else '365nm', 'uv'))
    for name, suffix in required:
        filter_id = ids_by_name.get(name.casefold())
        if filter_id is None or filter_id not in slots:
            raise ValueError(
                f"The {name} filter must be assigned to a filter-revolver slot."
            )
        targets.append(
            FilterCaptureTarget(
                name=name,
                suffix=suffix,
                position=slots.index(filter_id) + 1,
            )
        )
    return targets


def capture_series_stem(target_folder: str) -> str:
    """Return the final folder name used as the image-set filename stem."""
    stem = os.path.basename(os.path.normpath(target_folder))
    if not stem or stem in (os.path.sep, os.path.altsep):
        raise ValueError("The save location must have a final folder name.")
    return stem


def capture_folder_is_empty(target_folder: str) -> bool:
    """Return whether the selected save folder has no files or subdirectories."""
    with os.scandir(target_folder) as entries:
        return next(entries, None) is None


def next_capture_series_index(target_folder: str, stem: str) -> int:
    """Choose a monotonic set index without overwriting any B/G/R series image."""
    pattern = re.compile(
        rf"^{re.escape(stem)}_(\d+)_(?:(?:uv255|uv310|uv365|vis)_)?(?:[bgr]|uv)\.jpg$",
        flags=re.IGNORECASE,
    )
    highest_index = 0
    with os.scandir(target_folder) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match:
                highest_index = max(highest_index, int(match.group(1)))
    return highest_index + 1


def capture_filename(stem: str, series_index: int, suffix: str, channel: str | None = None) -> str:
    """Build a filename without its JPEG extension for the shared save helper."""
    if series_index < 1:
        raise ValueError("Capture series indices start at 1.")
    if suffix not in {"b", "g", "r", "uv"}:
        raise ValueError("Capture suffix must be b, g, r, or uv.")
    if channel is not None and channel not in ('uv255', 'uv310', 'uv365', 'vis'):
        raise ValueError("Capture channel is invalid.")
    channel_part = f"_{channel}" if channel else ""
    return f"{stem}_{series_index}{channel_part}_{suffix}"
