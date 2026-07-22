# TabletScanner AI working guide

This file applies to the entire repository. It is the AI-oriented companion to
[README.md](README.md), which contains the product and architecture overview.

The active four-channel illumination and settings redesign is specified in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Resolve its listed hardware decisions before
implementing command mappings or UV behavior.

## Mission

Make UI and backend changes as coordinated full-stack changes while preserving physical-device
safety, operator data, and the Windows packaging path. Prefer adding clear component/service
boundaries over extending the largest existing files.

Current code is the ultimate source of truth. `.github/copilot-instructions.md` predates the
Electron and recipe-pipeline work and contains stale assertions about route count, Flask
threading, background threads, serial details, and frontend structure. Do not repeat those
assertions without checking the current source.

## Read before changing code

Start with `README.md`, then inspect the smallest relevant set:

| Change | Read first |
| --- | --- |
| App layout/navigation | `frontend/src/app/app.component.ts`, `.html`, `.css`, `shared.service.ts` |
| Scanner UI | The affected file under `frontend/src/app/features/`, plus every caller of its API endpoint |
| Shared API/error behavior | `api-config.ts`, `app.config.ts`, `popup.interceptor.ts`, `error-notification.service.ts` |
| Pipeline UI | `models/pipeline.models.ts`, `recipe.service.ts`, `pipeline-state.service.ts`, affected recipe components |
| Flask endpoint | Route and helpers in `backend/app.py`, corresponding frontend call sites, `error_codes.py` |
| Camera | `cameracontrol.py`, `globals.py`, all `grab_lock` call sites, affected routes |
| Motion/lights | `porthandler.py`, `motioncontrols.py`, `globals.py`, affected routes and UI controls |
| Autofocus/measurement | `autofocus_main.py`, `app.py`, `auto-measurement.service.ts`, auto-measurement component |
| Pipeline step | `pipeline_types.py`, `pipeline_steps.py`, `pipeline_validators.py`, `pipeline_engine.py`, relevant `proc_elements/` |
| Persistence | `settings_manager.py`, `recipe_manager.py`, `calibration_manager.py`, packaging behavior in `build.bat` |
| Desktop bridge/package | `electron/main.js`, `electron/preload.js`, `electron/package.json`, `build.bat` |

Search for endpoint names, error codes, setting keys, step IDs, and JSON fields across both
`frontend/` and `backend/` before changing them. Several contracts are coupled by string rather
than by generated types.

## Repository boundaries

### Source files

- Active Angular root: `frontend/src/main.ts` → `app.component.ts`. The parallel
  `app.ts`/`app.html`/`app.scss` starter root is unused.
- Active navigation: section state in `SharedService`. `app.routes.ts` is empty and
  `app-routing.module.ts` is legacy.
- Backend entry: `backend/app.py`.
- Pipeline catalog/runtime authority: `backend/pipeline_steps.py` and `pipeline_types.py`.
- Frontend pipeline contract mirror: `frontend/src/app/models/pipeline.models.ts`.
- Desktop source: `electron/main.js`, `preload.js`, and `package.json`, even though the root
  ignore rule currently hides the entire directory.

### Generated, vendor, and runtime output

Do not hand-edit or review as source:

- `frontend/node_modules/`, `frontend/dist/`, `frontend/.angular/`, coverage output
- `backend/build/`, `backend/dist/`, `__pycache__/`, `*.pyc`
- `electron/node_modules/`, `electron/app/`
- `release/`
- `backend/tabletscanner_backend.log*`
- ignored full firmware/archive output

The Electron directory mixes source and generated files. If Electron work is in scope, edit only
its explicit source/config files and call out the root `.gitignore` defect.

### Mutable tracked data

Treat these as user/runtime data, not convenient test fixtures:

- `backend/settings.json`
- `backend/recipes/*.json`
- `backend/calibrations/calibrations.json`
- `backend/a2A4508-20ucBAS_40697387.pfs`

Do not overwrite or normalize them during tests. Use temporary directories/files. Inspect diffs
carefully because normal application use can modify the tracked JSON files.

## Hardware safety rules

### Motion and lights

- Serialize serial I/O with `porthandler.motion_lock`. Prefer
  `porthandler.write_and_wait()` or `write_and_wait_motion()`; use raw `write()` only where the
  existing protocol deliberately waits elsewhere.
- Keep finite serial timeouts. Current discovery uses 115200 baud, 0.2 s read timeout, 0.5 s
  write timeout, USB `0483:5740`, `M115`, and the mapping marker
  `TS-LIGHT-V3-P0F0-P1F1-P2HE0-P3HE1-LOCK`.
- Preserve backend clamping and cached position updates. Current limits are X/Y 0–175 mm and Z
  0–30 mm. UI limits are not an adequate safety boundary.
- Respect `globals.motion_busy`; status polling must not interleave commands with homing,
  autofocus, or long motion. The flag currently covers homing incompletely and is not a complete
  operation-ownership mechanism; extend coordination deliberately when touching these flows.
- Preserve disconnect cleanup in both `globals.motion_platform` and
  `porthandler.motion_platform` until ownership is intentionally consolidated.
- Maintain the four-channel light interlock and configured automatic shutoff: each UV channel has
  independent dimmed/full timeouts; VIS has no thermal timeout. Error and abort paths must make a
  best effort to turn every configured output off.
- Never add a hardware command to a GET endpoint. New mutating actions use POST and return only
  after command acceptance/completion semantics are clear.

### Camera

- Use `globals.grab_lock` around acquisition/stream state transitions that can overlap.
- Use `cameracontrol.grab_and_convert_frame()` for new frame consumers. Its returned array is a
  copied BGR8 frame; do not add a second Bayer conversion.
- Check handles for actual liveness, not only non-nullness. Preserve cleanup of grabbing and
  open state on disconnect.
- Do not let preview, live stream, autofocus, and capture grab concurrently without an explicit
  arbitration design.
- Keep hardware-free failures recoverable. Do not replace a missing device with silent fake
  success in production paths.

### Concurrency

The backend is not globally single-threaded: Flask can handle concurrent requests and the lamp
monitor is a daemon thread. Module globals and JSON caches are shared mutable state. Use the
existing serial, camera, settings, recipe, and calibration locks; do not assume request order.

## Full-stack contract rules

### REST API

- The current base is `http://localhost:5000/api` in `frontend/src/app/api-config.ts`.
- Keep endpoint request/response shapes explicit in TypeScript interfaces and backend code.
  Avoid introducing more `any`, response-message substring parsing, or filename-based state.
- Preserve HTTP status semantics. Do not return HTTP 200 with an error-shaped body for a new
  failure path.
- Existing errors are inconsistent, but new actionable errors should use
  `{ error, code, popup }`. Add stable constants to `backend/error_codes.py`.
- The interceptor resolves messages from `frontend/src/assets/error_messages.json`, not from the
  backend text. Keep that file synchronized with `backend/error_messages.json` when scanner
  error mappings change.
- Search all call sites before renaming routes. Known mismatch: the motion UI posts to
  `/api/disable_steppers`, but no Flask route currently implements it.
- Keep the API bound to local desktop use unless authentication, CORS, and filesystem endpoint
  security are designed together.

### Settings and persistence

- `settings_manager` caches a dictionary and persists it beside `app.py` in development or
  beside the frozen executable in packaged mode.
- Validate setting category, key, type, and safe range on the backend. UI validation is only a
  convenience.
- Do not embed new developer-specific absolute paths in defaults.
- Recipe documents use `schema_version: 1` and snake_case. If the schema changes, define a
  migration/backward-compatibility plan before writing new files.
- Recipe/calibration managers currently derive storage from `__file__`, unlike settings. Any
  packaging/storage refactor must account for writable installed locations and migration of
  existing operator data.
- Backend pipeline data types and auto-conversions are currently descriptive: sequential type
  compatibility is not enforced or converted by the backend validator/engine. The frontend has
  separate compatibility rules. Do not assume either side protects the other.

### Pipeline steps

When adding or changing a step, update every relevant layer:

1. Put the image/data operation in `backend/proc_elements/` when it is independently reusable.
2. Register one `StepDefinition` and executor in `backend/pipeline_steps.py` with stable ID,
   input/output types, parameter schema, side outputs, and prerequisites.
3. Update validation and conversion logic in `pipeline_validators.py`, `pipeline_engine.py`, and
   `pipeline_types.py` if the data contract changes.
4. Update the TypeScript model only when the catalog/document wire schema changes. The toolbox
   itself should remain backend-catalog driven.
5. Inspect `PipelineStateService` for type overrides, aggregation rules, secondary inputs, and
   preview behavior tied to particular step IDs.
6. Inspect the inspector/preview for step-specific controls and side-output rendering. Update
   `frontend/src/assets/node-descriptions.json` for operator help.
7. Test validation, full execution, partial preview, empty input, multiple images, and invalid
   parameters. Use small generated arrays or temporary image files.

Do not register only a UI node or only a Python executor; that creates a pipeline document that
cannot round-trip or run.

## UI implementation rules

- Keep operator-facing text consistent with the existing Hungarian UI unless the task includes
  localization. Preserve domain terms already used in saved data/API fields.
- Prefer typed feature services for new backend calls. Existing direct `HttpClient` use in large
  components is debt, not a pattern to expand.
- Put cross-feature state behind a service with clear ownership. Keep purely visual state local.
- Clean up intervals, subscriptions, event listeners, object URLs, and imperative DOM elements
  in `ngOnDestroy`. Section switching destroys and recreates feature components.
- Preserve desktop behavior, but do not add more fixed geometry without a hardware/config
  reason. The current UI is tightly coupled to a maximized window and 10×10 tray.
- Maintain keyboard focus and accessible labels for new controls. Global styles currently remove
  focus outlines; do not rely on that behavior.
- `step-inspector.component.ts` and `pipeline-preview.component.ts` are already several thousand
  lines. Extract new cohesive UI into standalone components/templates/styles instead of adding
  another large inline block.
- Do not modify the unused Angular starter root or legacy routing module to implement an active
  feature.

## Backend implementation rules

- `backend/app.py` is already route-heavy. For major new capabilities, prefer a focused service
  module and, when practical, a Flask Blueprint rather than adding all business logic inline.
- Separate HTTP parsing/serialization from device or processing logic so the latter can be
  tested without starting Flask or connecting hardware.
- Use `logging`, not `print`, in application code. Avoid logging raw image arrays, binary data,
  or sensitive local paths at routine levels.
- Bound loops, retries, subprocess calls, and hardware waits. Expose cancellation for operations
  that can block the operator workflow.
- Use context managers and `finally` cleanup for temporary files, light state, motion flags,
  camera grabs, and external processes.
- Keep frozen-mode behavior in mind: Electron sets `TABLETSCANNER_FULL=1`; standalone frozen
  mode intentionally restricts routes.

## Electron and packaging rules

- Keep `contextIsolation: true` and `nodeIntegration: false`. Expose only narrow, validated IPC
  operations through `preload.js`.
- Validate renderer-provided paths and payloads in the main process before filesystem writes.
- The Electron dev command consumes compiled Angular and a frozen backend; it is not equivalent
  to running the browser UI and Python source.
- `build.bat` replaces generated directories and `release/`. Do not run the full packaging build
  merely for a frontend unit change unless packaging validation is needed.
- If packaging inputs change, verify the PyInstaller data list, Angular output path/base href,
  Electron `files`, copied backend resources, runtime writable data, and startup/shutdown.
- Before relying on a clean clone, fix or explicitly account for the root rule that ignores all
  Electron source.
- Electron currently terminates the backend process directly on exit, so Python `finally`
  cleanup is not guaranteed on Windows. Changes to shutdown, lights, or device ownership must
  test the packaged exit path.

## Known sharp edges

- `backend/app.py` combines roughly 59 API routes with hardware, orchestration, filesystem, and
  pipeline responsibilities.
- `backend/globals.py` creates a second Flask object; `cameracontrol.py` imports it for logging,
  while requests use the app from `app.py`. Do not register routes or request state on the wrong
  object.
- Motion helpers swallow some non-disconnect exceptions, while callers can still update cached
  positions or report success. Do not build new behavior on those success assumptions.
- Health is liveness only. `backend_ready` is unused, and `/api/health` does not report device
  readiness.
- Long autofocus, measurement, pipeline, montage, and dialog operations run synchronously in
  request workers. There is no general job ownership/progress/cancellation model.
- The API has unrestricted CORS, raw G-code, and caller-supplied local read/write/open/delete
  paths. It is a trusted localhost API, not a remotely safe service.
- JSON writes are not atomic and runtime data does not yet have one versioned, writable app-data
  location.
- Packaged native-dialog helpers and recipe/calibration/profile resources are incomplete; test
  them from the frozen application, not only Python source.
- Preview requests can race in the recipe UI because prior HTTP calls are not cancelled. Avoid
  allowing an older response to overwrite newer parameter state.

## Validation commands

Run the smallest relevant checks, then widen for cross-layer changes.

```powershell
# Backend smoke test (standalone assertions, not pytest)
cd backend
$env:PYTHONUTF8 = '1'
python test_select_channel.py

# Frontend production compile
cd ..\frontend
npm run build

# Frontend unit tests
npm test -- --watch=false --browsers=ChromeHeadless

# Full Windows package, only when relevant
cd ..
.\build.bat
```

Record baseline versus new failures. At the time this guide was created, the backend smoke test
passes, while the frontend production build fails its component-style budget and all ten
headless specs fail due to existing test setup/starter-test problems. Do not claim a green test
suite until those failures are repaired.

For backend changes without hardware, add focused tests around pure helpers, pipeline execution,
Flask test-client contracts, or mocked adapter boundaries. Do not require a physical camera or
motion platform for ordinary CI tests.

## Definition of done for major UI/backend work

- The user flow and ownership boundary are clear.
- Frontend and backend request/response models agree, including failure and cancellation paths.
- Hardware operations remain locked, bounded, clamped, interruptible where applicable, and safe
  on disconnect.
- Mutable operator data is preserved or migrated deliberately.
- New code is extracted from monoliths where it introduces a distinct responsibility.
- Tests cover the changed seam without hardware; relevant manual hardware checks are listed.
- Production Angular build and relevant tests were run, with pre-existing and new failures
  distinguished.
- Packaged mode was considered; it was exercised when startup, paths, IPC, frozen data, or build
  inputs changed.
- `README.md` and this guide are updated if architecture, setup, persistence, or baseline facts
  changed.
