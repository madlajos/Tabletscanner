# Tabletscanner AI Coding Guidelines

## Project Architecture

**Tabletscanner** is a full-stack application for tablet/document scanning with hardware motion control and camera autofocus. It consists of three distinct layers:

- **Backend** (`backend/`): Python Flask API (port 5000) managing camera (Basler Pylon), motion platform (serial), and autofocus
- **Frontend** (`frontend/`): Angular 20 SPA communicating with backend via REST API
- **Firmware** (`Firmware/`): Marlin firmware for the motion platform (3D printer-based hardware)

### Critical Data Flow

1. **Camera Stack**: Basler Pylon camera → frame buffer (locked) → BayerGR10p→BGR8 conversion → OpenCV processing → JPEG stream
2. **Motion Control**: Flask HTTP → serial port (115200 baud, 0.2s timeout) → G-code commands (Marlin) → hardware feedback
3. **Autofocus**: Real-time frame capture → edge detection → algorithm calculates Z-axis movement → motion platform repositions

## Backend Patterns

### Global State Management (`globals.py`)

Device state is stored in module-level variables with threading locks:
- `camera`: Basler Pylon camera object (must check `IsOpen()` and `IsGrabbing()`)
- `motion_platform`: Serial port device (use `porthandler.motion_lock` for concurrent access)
- `stream_running`: Boolean flag for video stream lifecycle
- `motion_busy`: Set when motion operations in progress (prevents concurrent commands)

**Key rule**: Always use the RLock in `porthandler.motion_lock` when reading/writing to motion_platform; never access the serial port directly.

### Serial Communication (`porthandler.py`)

Pattern for serial device connection:
```python
def connect_to_serial_device(device_name, identification_command, expected_response, vid, pid):
    # 1. Scan available COM ports by VID/PID (e.g., vid=0x0483, pid=0x3748 for STM32)
    # 2. Send short command (e.g., "M115") with ~1s timeout (never block Flask thread)
    # 3. Check response for expected string (e.g., "Marlin")
    # 4. Return serial.Serial object or None
```

**Timeouts are critical**: Always use 0.2s read/write timeouts. Flask is single-threaded by default; blocking calls hang the UI.

### Flask API Error Handling

All endpoints return JSON with a consistent error structure:
```python
{
    'error': 'Human-readable error message',
    'code': 'E1111',  # ErrorCode constant
    'popup': True     # Signals frontend to show popup notification
}
```

Error codes defined in `error_codes.py`, messages in `error_messages.json`. Example: `ErrorCode.CAMERA_DISCONNECTED = "E1111"`.

### Camera & Frame Handling (`cameracontrol.py`)

All frame operations (live streaming, image capture, autofocus) use a unified grab-and-convert pattern:

```python
def grab_and_convert_frame(camera, timeout_ms=5000) -> np.ndarray:
    """Grab frame from camera and convert BayerGR10p -> BGR8 immediately."""
    # Returns: uint8 BGR8 array (HxWx3), already copied and safe to use
```

This centralized function ensures:
- **Immediate Bayer→BGR conversion**: All downstream code works with BGR8
- **Lock safety**: Frame is copied before release, safe to use outside lock
- **Consistent behavior**: Stream, image saving, and autofocus all use the same code path

The global `converter` object handles BayerGR10p → OpenCV BGR8 conversion. Frame streaming uses a lock pattern to prevent concurrent access:
```python
with globals.grab_lock:
    frame_bgr = grab_and_convert_frame(cam, timeout_ms=5000)
    # Frame is already BGR8 and copied; safe to use outside lock
```

**Important**: All frames returned from grab operations are already BGR8 and copied. No manual conversion needed.

### Settings Persistence (`settings_manager.py`)

Settings are loaded at startup, cached in `_cached_settings`, and protected by `_settings_lock`:
```python
load_settings()  # Reads settings.json
save_settings()  # Writes _cached_settings to disk
get_settings()   # Returns in-memory dict (read-only for safety)
```

## Frontend Patterns

### Service Architecture

- **Services** in `src/app/services/`: Each service wraps HTTP calls to a specific backend subsystem
  - Auto-measurement, camera control, motion control, etc.
  - Use dependency injection; services are singletons
- **Features** in `src/app/features/`: Standalone Angular components (Angular 20 no module-based architecture)

### HTTP Communication

All API calls use `HttpClient` from `@angular/common/http`. Backend is at `localhost:5000/api/` (hardcoded or via environment config).

Expected response format:
```typescript
// Success
{ /* endpoint-specific data */ }

// Error
{ error: string, code: string, popup?: boolean }
```

### Error Notification

The `error-notification.service.ts` and `ErrorPopupListComponent` listen for error events and display popup alerts to the user.

## Development Workflows

### Running the Full Stack

1. **Backend**: `python backend/app.py` (requires pypylon, opencv, flask)
2. **Frontend**: From `frontend/` directory, run `npm install && ng serve` (serves on `localhost:4200`)
3. **Hardware**: Motion platform must be connected via USB serial (auto-detected by VID/PID)

### Testing Serial Connections

- Use a serial monitor (e.g., PuTTY) to test G-code commands directly
- Common commands: `M115` (identify), `M114` (position), `G28` (home), `M84` (disable steppers)
- Response format: Plain text, ends with "ok" or "error"

### Debugging Camera Issues

Enable debug logging in `logger_config.py` (DEBUG level). Basler Pylon errors are device-specific; check camera via Basler Pylon Viewer before troubleshooting in app.

### Building for Distribution

Backend can be frozen as `.exe` using PyInstaller. Path resolution in `logger_config.py` and `settings_manager.py` checks `sys.frozen` to support this.

## Cross-Component Communication Patterns

### Motion Platform State

Endpoint `/api/get_motion_platform_position` returns cached position (`globals.last_toolhead_pos`) when `motion_busy=True`. This prevents UI blocking during long operations (homing, autofocus). Always check `motion_busy` before issuing new motion commands.

### Autofocus Algorithm (`autofocus_main.py`)

The coarse autofocus runs in the Flask thread, not a background worker:
1. Iteratively moves Z-axis (small increments)
2. Captures frame via `acquire_frame()` (uses unified `grab_and_convert_frame()`)
3. Processes with `detect_largest_object_square_roi()` on BGR8 frame
4. Moves to next position
5. Returns final position when focused

All frames are BGR8 immediately after grabbing. **Warning**: This blocks the Flask thread for ~5-10 seconds. No concurrent camera/motion operations allowed during autofocus.

### Settings Updates

When frontend modifies camera or motion settings, the backend validates via `validate_and_set_camera_param()` and immediately persists to `settings.json`. Frontend listens via polling or WebSocket (if implemented) to sync.

## Project-Specific Conventions

- **Naming**: Underscore_case for Python, camelCase for TypeScript
- **Logging**: Use `app.logger` in backend; avoid print() statements
- **Constants**: Stored in `globals.py` (motion limits), `error_codes.py` (error codes), settings JSON (user-configurable)
- **Firmware**: Marlin flavor (not Klipper or RepRap); G28, M84, M114 are standard commands
- **No background threads**: Flask runs single-threaded by design; use locks for shared state, avoid threads unless explicitly needed

## Key Files Reference

- **Backend entry**: [backend/app.py](../backend/app.py) (Flask app, 26 API routes)
- **Camera control**: [backend/cameracontrol.py](../backend/cameracontrol.py) (Basler Pylon wrapper)
- **Motion control**: [backend/motioncontrols.py](../backend/motioncontrols.py) (G-code commands), [backend/porthandler.py](../backend/porthandler.py) (serial comms)
- **Autofocus algorithm**: [backend/autofocus_main.py](../backend/autofocus_main.py)
- **Global state**: [backend/globals.py](../backend/globals.py), [backend/settings_manager.py](../backend/settings_manager.py)
- **Frontend app**: [frontend/src/app/app.component.ts](../frontend/src/app/app.component.ts) (main container)
- **Frontend services**: [frontend/src/app/services/](../frontend/src/app/services/) (HTTP wrappers)