# Four-channel illumination and scanner settings implementation plan

Status: **Draft for approval**  
Scope: BTT Octopus V1.1 migration, four-channel light control, scanner settings UI changes,
global camera exposure/gamma, software settings modal, and auto-measurement capture plans.  
Audience: future AI agents and developers implementing the work in this repository.

This plan must be read together with [AGENTS.md](AGENTS.md) and [README.md](README.md). Check off
tasks only after their acceptance criteria and relevant tests pass. Do not infer hardware
commands that are not recorded in this plan or an approved hardware-protocol document.

## 1. Intended outcome

The scanner moves from the current two logical lights (`dome` and `bar`) on a BTT SKR Mini E3
V3 to four independently addressable channels on a BTT Octopus V1.1:

| Canonical ID | Operator label | Octopus output | Safety class |
| --- | --- | --- | --- |
| `uv255` | `255 nm` | `HE0` | UV |
| `uv310` | `310 nm` | `HE1` | UV |
| `uv365` | `365 nm` | `HE2` | UV |
| `vis` | `VIS` | `FAN0` | Visible |

The UI will provide four manual lamp buttons. Camera exposure and gamma become global camera
settings rather than per-light settings. Automatic measurement will execute an ordered capture
plan whose rows select a wavelength and filter position.

The safest default design is **one active light channel at a time**. Activating a channel first
turns off all other channels and updates software state only after the board acknowledges the
commands.

## 2. Decisions required before implementation

Hardware implementation must stop at the affected phase if these decisions are unresolved.

| ID | Required decision | Why it matters | Provisional plan |
| --- | --- | --- | --- |
| D1 | **Confirmed:** the first wavelength is **255 nm**, never 240 nm. | A mislabeled UV channel is a safety and data-integrity defect. | Use canonical `uv255` and display/persist `255 nm` everywhere. |
| D2 | Exact custom-firmware G-code channel selectors for `HE0`, `HE1`, `HE2`, and `FAN0`, plus OFF commands and acknowledgement text. | The current board uses `M106 P0 S255` and `M106 P3 S255`, but those selectors must not be assumed to map to Octopus headers. | Operators configure selector-only values such as `P0` in `Haladó`; the backend will form `M106 P<selector-number> S<pwm>` only after physical mapping and OFF/acknowledgement behavior are verified. PWM is `round(255 * percent / 100)`; 10% is `S26`, 100% is `S255`. |
| D3 | **Confirmed:** USB identity is `0483:5740`; `M115` reports `FIRMWARE_NAME:Marlin bugfix-2.0.x`, machine type `Tablet Scanner`, and acknowledges with `ok`. | `porthandler.py` currently detects `0483:5740` and expects Marlin. | Keep the existing 115200/Marlin discovery rule. The successful USB-only check is recorded in `docs/HARDWARE_LIGHT_PROTOCOL.md`; repeat it with the final powered installation before release. |
| D4 | **Confirmed design:** each UV channel has operator-configured dimmed/full brightness percentages and separate dimmed/full auto-off times. The initial numeric values are still required. | UV timeout is a thermal-safety requirement and must match the active power mode. | Persist validated values in the `Lámpa` settings page; reject UV activation if its configuration is absent/invalid. |
| D5 | **Confirmed:** channels are mutually exclusive. | Simultaneous channels affect power, optical results, heat, and safety. | Enforce mutual exclusion in firmware/backend/UI. Auto measurement always captures sequentially. |
| D6 | **Confirmed:** assets use `255nm_on.svg`/`255nm_off.svg` and the matching `310nm`, `365nm`, and `vis` basename pattern. | The manual controls need a stable asset contract. | Centralize the names in the frontend light definition and verify the supplied files, including filename case, during Phase 4. |
| D7 | **Confirmed:** `VIS` appears in the `Lámpa` table but has only a single-click, full-brightness control and no thermal auto-off requirement. | VIS must be visibly documented without incorrectly exposing UV dim/full or thermal-time controls. | Render a read-only VIS row with `N/A` for dimmed brightness and both cutoff-time columns; its full-brightness value is fixed at 100%. |
| D8 | Which UV brightness mode should automatic measurement use? | Unattended full-power UV capture needs an explicit thermal and optical decision. | Until confirmed, automatic UV captures use dimmed mode; VIS captures use its normal full mode. |

The filter changer was subsequently extended with manual physical control. Positions 1–6 are
selected, validated, persisted, sent to the backend, and included in capture metadata. Manual
control homes the firmware's external `A` axis (internal Marlin `I` axis) and performs acknowledged
60° steps. Automatic capture-plan execution still treats its filter position as metadata only.

## 3. Target contracts

Use these canonical identifiers across TypeScript, JSON, Python, API payloads, filenames, and
logs. Do not introduce additional `dome`, `bar`, `lamp_top`, or `lamp_side` logic.

### 3.1 Shared light model

```typescript
export type LightChannel = 'uv255' | 'uv310' | 'uv365' | 'vis';
export type UvBrightnessMode = 'dimmed' | 'full';

export interface UvLampSettings {
  dim_percent: number;           // numeric 10..100; UI renders a % suffix
  full_percent: number;          // numeric 10..100; UI renders a % suffix
  dim_timeout_seconds: number;   // numeric seconds; UI renders an s suffix
  full_timeout_seconds: number;  // numeric seconds; UI renders an s suffix
}

export interface LightStatus {
  active_channel: LightChannel | null;
  active_mode: UvBrightnessMode | null;
  channels: Record<LightChannel, boolean>;
  auto_turned_off: LightChannel[];
}
```

The backend should define the equivalent Python enum/dataclass or validated constants. A single
channel definition must own:

- canonical ID and Hungarian/operator label;
- physical output (`HE0`, `HE1`, `HE2`, `FAN0`);
- approved ON/OFF G-code;
- UV flag and timeout;
- optional settle time;
- frontend ON/OFF asset names.

Frontend code may mirror display data, but physical commands and safety policy belong only to
the backend.

### 3.2 Typed light API

Add typed endpoints and stop using raw G-code from lamp buttons:

```text
GET  /api/lights/status
POST /api/lights/activate   { "channel": "uv310", "mode": "dimmed" }
POST /api/lights/off        { "channel": "uv310" }  # channel optional; omitted means all
POST /api/turn-off-all-lights                         # retained compatibility/safety alias
GET  /api/settings/lamp
PUT  /api/settings/lamp    { "channels": { "uv255": { ...UvLampSettings } } }
```

Every mutation must:

1. validate the channel;
2. acquire operation/serial ownership;
3. turn off conflicting channels;
4. send the approved command with a finite timeout;
5. update state/timer only after acknowledgement;
6. select the configured UV percentage for the requested mode, calculate its PWM as
   `round(255 * percent / 100)`, and send the approved board command;
7. return typed state or the standard `{error, code, popup}` envelope.

`mode` is required for UV activation and must be omitted for VIS. The backend treats VIS as fixed
full brightness and does not start a thermal auto-off timer for it.

`GET /api/lights/status` must not consume a one-shot global flag in a way that makes multiple UI
consumers race. Return current state plus auto-off events/state in a repeatable form.

### 3.3 Global camera settings

Replace the two persisted camera sections with one schema:

```json
{
  "settings_schema_version": 2,
  "camera_params": {
    "ExposureTime": 100000.0,
    "Gamma": 1.0
  }
}
```

`Gain` is not part of the new editable/persisted UI contract. A `.pfs` camera profile may still
configure other hardware properties, but switching light channels must never change exposure,
gamma, or gain.

The camera API should return and update this single typed object. The implementation may keep
the existing route names to limit churn, but `/api/update-camera-settings` must become a real,
validated update endpoint instead of applying a setting and then returning a deprecation error.
Remove `/api/update-camera-settings-light` after all callers are migrated.

### 3.4 Auto-measurement capture plan

```typescript
export interface CapturePlanRow {
  id: string;                    // UI-only stable row identity
  wavelength: LightChannel;
  filter_position: 1 | 2 | 3 | 4 | 5 | 6;
}

export interface CaptureRequestRow {
  wavelength: LightChannel;
  filter_position: number;
}
```

Replace `lamp_top` and `lamp_side` in `TabletStepRequest` with:

```json
{
  "capture_plan": [
    { "wavelength": "vis", "filter_position": 1 },
    { "wavelength": "uv365", "filter_position": 3 }
  ]
}
```

Return structured image information instead of requiring filename parsing:

```json
{
  "saved_images": [
    {
      "path": "...",
      "wavelength": "uv365",
      "filter_position": 3,
      "masked": false
    }
  ]
}
```

Persist the last valid capture plan under `auto_measurement_settings.capture_plan` so it survives
restart. Default to one non-deletable row using `VIS` and filter `1`, which is the safest initial
state. Duplicated rows are allowed because repeated captures may be intentional.

The capture-plan table does not gain a brightness column in this scope. Until D8 is explicitly
changed, a UV capture activates its channel in `dimmed` mode and uses that mode's timeout.

## 4. Phased implementation

### Phase 0 — establish the implementation baseline

- [ ] Record the confirmed D1, D4, and D5 decisions and resolve remaining D2, D3, D6–D8.
- [ ] Add `docs/HARDWARE_LIGHT_PROTOCOL.md` containing the approved board identity, header map,
      G-code, power values, acknowledgements, boot state, and emergency OFF sequence.
- [ ] Confirm the Octopus custom firmware starts with all four outputs OFF and keeps them OFF
      until an explicit command.
- [ ] Add/repair focused Angular test setup for the active standalone `AppComponent` and scanner
      components; remove or repair tests that target the unused starter `App`.
- [ ] Decide how to handle the existing Angular style-budget failure. Do not hide it by silently
      increasing budgets as part of this feature.
- [ ] Add `requests` to `backend/requirements.txt` and define the Python/Node versions used for
      this work.
- [ ] Fix or explicitly remove the existing `/api/disable_steppers` UI call while the controller
      integration is being touched.
- [ ] Ensure Electron source is tracked before packaged acceptance testing.

Gate: software-only model/UI work may begin before the physical board arrives. Hardware command
code and hardware acceptance tests require D2–D4 and the approved protocol document.

### Phase 1 — introduce domain models and settings migration

Primary files:

- `backend/settings_manager.py`
- `backend/settings.json`
- new `backend/light_control.py`
- `frontend/src/app/shared.service.ts`
- new `frontend/src/app/models/light.models.ts`
- `frontend/src/app/services/settings-updates.service.ts`
- `frontend/src/app/services/auto-measurement.service.ts`

Tasks:

- [ ] Add canonical light-channel definitions and typed frontend models.
- [ ] Change `SavedImageInfo.lightType` from `'dome' | 'bar'` to `LightChannel`; add
      `filterPosition` and `masked` where relevant.
- [ ] Keep active-light state for lamp/gallery behavior, but remove `lightSettingsSubject` and
      `applyCameraSettingsForLight`; lamp changes no longer trigger camera changes.
- [ ] Add `settings_schema_version: 2` and a tested migration in `settings_manager.py`.
- [ ] For existing files, initialize global `camera_params` from `camera_params_dome` because VIS
      replaces the old dome/reference illumination; fall back to `camera_params_bar` only if the
      dome section is absent. This migration rule must be confirmed before implementation.
- [ ] Retain only validated `ExposureTime` and `Gamma`; remove `camera_params_dome`,
      `camera_params_bar`, and obsolete preset-name state after a successful migration.
- [ ] Make settings writes atomic (temporary file plus replace) and preserve a one-time backup
      before schema migration.
- [ ] Add `auto_measurement_settings.capture_plan` with the safe default row.
- [ ] Add `lamp_settings.channels.uv255`, `.uv310`, and `.uv365`, each with numeric dim/full
      percentages and dim/full timeout seconds. Do not store `%` or `s` in JSON values. Keep VIS
      fixed at 100%, with no persisted thermal timeout.
- [ ] Validate brightness as 10–100 inclusive and timeout as a positive finite number before
      persisting or activating a UV channel.
- [ ] Do not overwrite real operator settings in tests; use temporary JSON files.

Acceptance:

- Old settings load without losing save path, objective, spacer-ring, background-subtraction, or
  camera-profile path.
- Restart produces the same schema-v2 values and does not repeatedly migrate.
- No runtime path reads `camera_params_dome` or `camera_params_bar` after migration.

### Phase 2 — build the Octopus light-control backend

Primary files:

- new `backend/light_control.py`
- `backend/porthandler.py`
- `backend/globals.py` (reduce old lamp globals)
- `backend/app.py` (thin routes/call sites only)
- `backend/error_codes.py` and both error-message JSON files
- custom firmware/configuration files when supplied

Tasks:

- [ ] Implement a `LightController` that owns channel definitions, active channel/mode, timers,
      and the fail-safe all-off sequence. Keep physical logic out of Flask routes.
- [ ] Use `porthandler.motion_lock` and acknowledged, bounded writes for every command.
- [ ] Update controller discovery for the approved Octopus USB identity and firmware response.
- [ ] Remove old connect-time `M106 P0/P3` commands from `porthandler`; invoke the controller's
      all-off sequence after a successful serial connection.
- [ ] Implement the typed light API from section 3.2.
- [ ] Replace the single dome/UV timestamps and high-power flag with per-channel active mode and
      deadlines. Select the deadline from the active UV channel's dimmed/full settings.
- [ ] Calculate UV PWM only in the backend with `round(255 * percent / 100)`; clamp/reject values
      outside 10–100 and never accept a client-supplied PWM value.
- [ ] Apply the selected strict timeout independently to `uv255`, `uv310`, and `uv365`. VIS has no
      thermal auto-off timer.
- [ ] On auto-off, command the physical channel OFF first, then report the new state/event.
- [ ] Make abort, measurement failure, disconnect, reconnect, shutdown, and Electron shutdown
      paths call the same all-off operation.
- [ ] Remove lamp-state parsing from `/api/send_gcode`. UI lamp control must not use this raw
      endpoint. Decide whether light-output commands should be rejected there to prevent state
      desynchronization.
- [ ] Replace `_turn_on_dome_light`, `_turn_on_uv_dome_light`, and `_turn_off_all_lights` internals
      with controller calls, then remove the old helpers once all callers are migrated.
- [ ] Use `vis` as the reference illumination for autofocus, empty-tablet checks, contour
      detection, and background subtraction unless a later optical requirement says otherwise.
- [ ] Apply the existing UV exposure gate to each UV capture, not to VIS.

Acceptance:

- Exactly one manual channel can be active.
- Failed/unacknowledged commands do not create false ON state.
- Each UV channel independently auto-switches off at the deadline for its confirmed active mode.
- All exit/error/abort paths attempt to switch off all four physical outputs.
- Tests use a fake serial device and verify exact command order; no physical hardware is required
  for unit/API tests.

### Phase 3 — simplify camera settings and persistence

Primary files:

- `backend/app.py`
- `backend/cameracontrol.py`
- `frontend/src/app/features/camera-control/camera-control.component.ts`
- `frontend/src/app/features/camera-control/camera-control.component.html`
- `frontend/src/app/features/camera-control/camera-control.component.css`
- `frontend/src/app/services/settings-updates.service.ts`
- `electron/main.js` and `electron/preload.js` if preset IPC becomes unused

Tasks:

- [ ] Make `apply_camera_settings()` read only `camera_params`.
- [ ] Make the camera settings GET return one typed `camera_params` object.
- [ ] Make global ExposureTime/Gamma updates validate, apply to an open camera, persist corrected
      hardware values, and return one consistent response.
- [ ] Remove `_apply_camera_settings_for_light` and every call from lamp, autofocus, manual-save,
      and automatic-measurement flows.
- [ ] Replace `cameraSettings: any` with the existing/updated `CameraSettings` interface.
- [ ] Remove `ExposureTime_Dome`, `ExposureTime_Bar`, `Gamma_Dome`, and `Gamma_Bar` fields and
      special matching/update logic.
- [ ] Remove lamp subscriptions from `CameraControlComponent`.
- [ ] Render one `Záridő` input and one `Gamma` input; remove the two icon columns and 2×2 grid.
- [ ] Keep the `Kamera Profil` row and camera connect/stream controls.
- [ ] Remove the `Beállítások` preset row and its `.tss` load/save methods. After repository-wide
      search confirms no caller remains, remove `save-tss-file` IPC from Electron and
      `settings_preset_name` from settings.
- [ ] If preset support is needed later, reintroduce it as an explicit page in the new software
      settings modal with a schema-v2 design; do not retain hidden legacy code.

Acceptance:

- Changing any lamp never changes camera parameters.
- Exposure and gamma apply identically under all four wavelengths.
- Camera profile selection, connect/disconnect, and stream controls still work.
- No `Dome`/`Bar` camera-setting keys, subscriptions, payloads, or preset fields remain.

### Phase 4 — implement four manual lamp buttons

Primary files:

- `frontend/src/app/features/motion-control/motion-control.ts`
- `frontend/src/app/features/motion-control/motion-control.html`
- `frontend/src/app/features/motion-control/motion-control.scss`
- new `frontend/src/app/services/light.service.ts`
- `frontend/src/app/shared.service.ts`
- `frontend/src/assets/SVG/` assets supplied for this work

Tasks:

- [ ] Add a typed `LightService` for status, activate, channel-off, and all-off calls.
- [ ] Replace `ringLightOn`/`uvDomeLightOn`, click/double-click timers, and raw `M106` calls with
      one `LightChannel` state model and backend-confirmed brightness mode.
- [ ] Render four buttons from a definition array in the required order: 255, 310, 365, VIS.
- [ ] For UV buttons, defer a single-click action briefly so a double-click cancels it: one click
      activates dimmed mode; a double-click activates full mode. A single click on the active UV
      channel turns it off. A double-click always selects full mode.
- [ ] Use the typed activate request with `mode: 'dimmed' | 'full'`; calculate neither PWM nor
      timeout in the frontend.
- [ ] Give VIS only the normal single-click toggle behavior; do not attach a double-click/full-mode
      handler or a thermal-timeout indicator.
- [ ] Use the supplied per-channel ON/OFF SVGs from the existing SVG folder.
- [ ] Preserve disabling during disconnect, measurement, homing where necessary, and autofocus.
- [ ] Poll or subscribe to repeatable backend status so timeout changes are reflected in every
      button without racing another consumer.
- [ ] Keep VIS as the automatically selected light for manual autofocus when no channel is on.
- [ ] When a channel activation fails, leave the previous/backend-reported visual state rather
      than optimistically displaying success.

Acceptance:

- Buttons show the backend-confirmed active channel, UV dimmed/full mode, and auto-off state.
- Activating one button deactivates all others.
- UI contains no hardware G-code or Octopus pin/header knowledge.
- Manual autofocus uses VIS and does not restore removed per-light camera settings.

### Phase 5 — rework collapsible scanner settings

Primary files:

- `frontend/src/app/features/camera-control/camera-control.component.ts`
- `frontend/src/app/features/camera-control/camera-control.component.html`
- `frontend/src/app/features/camera-control/camera-control.component.css`

Tasks:

- [ ] Reorder sections so `Kamerabeállítások` is first and `Mentési Beállítások` is last.
- [ ] Give each section independent local expanded/collapsed state; default both to expanded to
      preserve discoverability.
- [ ] Make the whole header a semantic button with `aria-expanded` and a visible expand/collapse
      indicator.
- [ ] Collapse only content, not the section header; do not destroy unsaved input state.
- [ ] Disable or preserve controls exactly as before during measurement/autofocus.
- [ ] Ensure keyboard Enter/Space and visible focus work even though global CSS currently weakens
      focus outlines.
- [ ] Keep layout usable at the packaged maximized window size and in narrower development
      windows.

Acceptance:

- Each header toggles only its own section using mouse and keyboard.
- `Mentési Beállítások` appears below camera settings.
- Collapse/expand does not trigger API writes or lose edited values.

### Phase 6 — add the software settings modal

Primary files:

- `frontend/src/app/app.component.ts`
- `frontend/src/app/app.component.html`
- `frontend/src/app/app.component.css`
- new `frontend/src/app/features/software-settings/` standalone components
- `frontend/src/app/app.config.ts` for Material dialog providers/imports if required

Tasks:

- [ ] Add a `Beállítások` button with a gear icon in the scanner's left panel immediately above
      the logo and above the logo's separator line.
- [ ] Implement the popup as a separate standalone modal/dialog component, not inline in
      `AppComponent`.
- [ ] Add left-side type navigation with entries `Szűrőváltó`, `Lámpa`, and `Tálca`; select the
      first type by default.
- [ ] Keep `Szűrőváltó` and `Tálca` as placeholders for now. Implement `Lámpa` as a live settings
      page, loaded from and saved through the typed lamp-settings API.
- [ ] In `Lámpa`, render one row each for 255 nm, 310 nm, 365 nm, and VIS with these columns:
      `Hullámhossz`, `Tompított fényerő`, `Teljes Fényerő`, `Tompított lekapcsolási idő`, and
      `Teljes fényerő lekapcsolási idő`.
- [ ] Use numeric text inputs with visible `%` suffixes for both brightness columns (10–100) and
      visible `s` suffixes for both timeout columns (positive seconds). Store numeric values only.
- [ ] Render the VIS row as read-only: `N/A` for dimmed brightness and both cutoff-time columns,
      and fixed `100%` for full brightness.
- [ ] Show inline Hungarian validation messages; do not save partial/invalid values. On a valid
      save, use the server-returned normalized settings as the UI source of truth.
- [ ] Prevent a settings update from changing an active UV channel's safety behavior: disable save
      while UV is active, or require the backend to turn all lights off before applying the update.
- [ ] Add close button, Escape close, backdrop close, focus trap/return, accessible dialog title,
      and sensible minimum/maximum sizing.
- [ ] Prevent opening or changing hardware settings during active measurement. Viewing may remain
      allowed only when no mutation controls are enabled.
- [ ] Do not add backend settings keys for placeholders until real fields and validation are
      specified.

Acceptance:

- The button is visually above the separator/logo and does not disturb the scanner grid.
- Left navigation changes the main pane; `Lámpa` loads and validates its persisted configuration.
- Dialog works by mouse and keyboard and restores focus to the gear button on close.

### Phase 7 — implement the auto-measurement capture-plan table

Primary files:

- `frontend/src/app/features/auto-measurement/auto-measurement.component.ts`
- `frontend/src/app/features/auto-measurement/auto-measurement.component.html`
- `frontend/src/app/features/auto-measurement/auto-measurement.component.css`
- `frontend/src/app/services/auto-measurement.service.ts`

Tasks:

- [ ] Keep only the `Autofocus` and `Háttérlevonás` toggle buttons; remove `Dóm fény` and
      `Súrló fény` state and markup.
- [ ] Add a table with `Hullámhossz`, `Szűrő`, and delete columns.
- [ ] Populate wavelength options from the canonical channel list; display `255 nm`, never
      `240 nm`.
- [ ] Populate filter options `1`–`6` from one constant.
- [ ] Initialize the first row as VIS/filter 1. Hide or disable its delete action so at least one
      row always remains.
- [ ] Add a `+` button below existing rows. New rows receive stable unique UI IDs and safe
      defaults.
- [ ] Allow deletion of every row except the first. Row order is capture order.
- [ ] Disable add/delete/dropdowns while measurement is active.
- [ ] Validate every row and require at least one row before enabling measurement.
- [ ] Persist valid rows to `auto_measurement_settings.capture_plan` with a short debounce or on
      explicit change; never write on every Angular change-detection pass.
- [ ] Snapshot and deep-copy the plan when measurement starts so in-flight work cannot change.
- [ ] Send `capture_plan` in every tablet request and remove `lamp_top`/`lamp_side`.
- [ ] Use structured response metadata to update the gallery; remove `_dome_` filename parsing.

Acceptance:

- First row cannot be removed; later rows can be added/deleted reliably.
- Dropdowns expose exactly the approved wavelengths and filter positions.
- Measurement start validates and snapshots the ordered plan.
- Reload restores the last valid plan, falling back safely when stored data is invalid.

### Phase 8 — execute capture plans in the backend

Primary files:

- `backend/app.py` (route orchestration only)
- new/extracted `backend/measurement_service.py`
- `backend/light_control.py`
- `backend/globals.py` latest-image state
- `backend/settings_manager.py`
- `frontend/src/app/features/image-viewer/image-viewer.component.ts`
- `frontend/src/app/shared.service.ts`

Tasks:

- [ ] Validate a non-empty `capture_plan`, known wavelength IDs, filter integer 1–6, row count,
      and payload size before any motion/light action.
- [ ] Extract the long per-tablet orchestration from `app.py` into a testable measurement service
      instead of adding another branch-heavy block to the route.
- [ ] Perform motion/autofocus once per tablet, then execute capture rows in order.
- [ ] For each row: record/select the placeholder filter position, activate the wavelength, wait
      its configured settle time, run the UV exposure gate when applicable, capture, attach
      wavelength/filter metadata, and turn the channel off in `finally`.
- [ ] Until D8 is changed, request `dimmed` mode for UV capture-plan rows and normal full mode for
      VIS rows. Full-power unattended UV capture must not be introduced implicitly.
- [ ] Use temporary VIS illumination for autofocus and contour/background-mask creation even if
      VIS is not a requested saved row; do not save an extra VIS image unless it is in the plan.
- [ ] Turn all channels off between rows unless the approved hardware protocol explicitly proves
      a safe optimized transition.
- [ ] Include wavelength and filter position in sanitized filenames and EXIF metadata, but treat
      structured response fields—not filenames—as the application contract.
- [ ] Replace fixed `latest_dome_*`/`latest_bar_*` globals with channel-keyed latest-image state.
- [ ] Extend latest-image endpoints to canonical channel IDs. Keep a temporary `dome -> vis`
      alias only if external consumers require it; do not guess a `bar` alias.
- [ ] Update manual capture to use the active canonical channel and global camera settings. If no
      light is active, require explicit VIS activation or return a clear error rather than
      silently labeling a capture as dome/VIS.
- [ ] Ensure stop, autofocus abort, quality failure, camera loss, serial loss, and unexpected
      exceptions all reach the common all-off cleanup.
- [ ] Keep capture-plan filter selection as metadata until measurement orchestration explicitly
      invokes the acknowledged manual filter-controller adapter and verifies each completed move.

Acceptance:

- One tablet produces one original image per plan row, in plan order, plus masked variants only
  when background subtraction is enabled.
- Every result identifies wavelength/filter without filename inference.
- UV exposure checks and timeouts apply to all three UV channels.
- No failure path leaves a channel logically or physically ON.
- Filter position is preserved as metadata and clearly identified as not physically actuated.

### Phase 9 — remove legacy paths and validate the release

Repository-wide cleanup searches must show no active use of:

```text
camera_params_dome
camera_params_bar
ExposureTime_Dome
ExposureTime_Bar
Gamma_Dome
Gamma_Bar
lamp_top
lamp_side
ringLightOn
uvDomeLightOn
_turn_on_dome_light
_turn_on_uv_dome_light
applyCameraSettingsForLight
```

Tasks:

- [ ] Remove dead CSS, imports, subscriptions, preset handlers, old SVG references, and obsolete
      Electron IPC after call-site searches.
- [ ] Decide and document compatibility behavior for legacy `dome`/`bar` latest-image clients and
      old auto-measurement payloads. Prefer an explicit migration error over silently mapping
      ambiguous UV data.
- [ ] Update `README.md`, `AGENTS.md`, API examples, settings schema, and hardware documentation.
- [ ] Run backend unit/API tests with fake serial/camera/filter adapters.
- [ ] Run Angular unit tests and production build; distinguish or resolve pre-existing failures.
- [ ] Build the Electron package and verify startup, modal assets, settings migration, runtime
      data location, backend shutdown, and all-off behavior.
- [ ] Test a copy of real operator settings and retain the pre-migration backup.

## 5. Required test matrix

### Backend software tests

- Channel validation and exact command mapping for all four channels.
- Mutual-exclusion command order and acknowledgement failures.
- Independent timeout behavior for 255/310/365 and VIS.
- All-off behavior on startup, reconnect, stop, abort, exception, and shutdown.
- Settings v1-to-v2 migration, invalid JSON, backup, atomic write, and idempotency.
- Lamp-settings validation, normalized round trip, missing configuration rejection, and PWM
  conversion boundaries (10% = `S26`, 100% = `S255`).
- Global ExposureTime/Gamma validation with camera open and disconnected.
- Capture-plan validation: empty, unknown wavelength, invalid filter, duplicates, and ordered rows.
- Auto-measurement sequencing with mocked motion, light, filter placeholder, and camera.
- Structured saved-image metadata and channel-keyed latest images.

### Frontend tests

- Four manual buttons render from definitions and follow backend state.
- UV single-click/double-click behavior selects dimmed/full mode correctly; the client never sends
  PWM values, and mode-specific timeout status clears the correct button.
- Timeout/status update clears the correct active button.
- Camera UI has one ExposureTime and one Gamma field and no light icons/preset row.
- Both scanner-setting sections collapse independently and remain accessible.
- Software settings modal navigation, close behavior, focus restoration, and the validated
  UV settings plus read-only VIS row in the `Lámpa` table.
- Capture-plan add/delete rules, stable IDs, options, validation, persistence, and measurement lock.
- Request payload contains a snapshot of `capture_plan`; gallery uses structured response data.

### Hardware acceptance tests

Perform initial tests without UV emitters connected, using a meter or safe dummy loads:

1. Confirm board identity and safe boot with HE0/HE1/HE2/FAN0 OFF.
2. Verify each approved G-code affects only its mapped output.
3. Verify mutual exclusion and all-off after failed/aborted operations.
4. Verify dimmed and full PWM output plus the corresponding approved timeout independently on
   every UV output.
5. Verify VIS single-click behavior, no thermal auto-off, and autofocus reference behavior.
6. Connect the real light under the project's UV safety procedure and repeat short-duration tests.
7. Run a two-row measurement and verify filenames, EXIF, response metadata, gallery, and physical
   output order. Treat filter position as metadata until actuator work is approved.
8. Exit the packaged Electron application while a channel is on and verify physical all-off.

## 6. Definition of done

The work is complete only when:

- confirmed decisions and D2/D3/D6/D8 are documented or explicitly accepted as release scope;
- four manual buttons control the intended Octopus outputs through typed backend APIs;
- all three UV channels have verified dimmed/full brightness, mode-specific strict auto-off, and
  all error paths fail safe;
- camera settings are global and no lamp switch changes camera parameters;
- camera/save sections collapse independently in the requested order;
- the gear-button modal provides the three settings types, including the persisted `Lámpa` table;
- auto measurement uses an ordered wavelength/filter table and structured request/response data;
- old two-light/per-light-camera code is removed or explicitly versioned for compatibility;
- software tests and the hardware acceptance matrix pass;
- packaged settings migration and shutdown/all-off behavior are verified;
- documentation reflects the final commands, schema, assets, and known limitations.
