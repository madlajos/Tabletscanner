# Door safety in the companion software

The supported identity is `TS-LIGHT-V3-P0F0-P1F1-P2HE0-P3HE1-LOCK-RHOME-DOORS1`.
The front Hall sensor is DIAG4/PG12; the top cover is DIAG5/PG13. Magnet present
means closed. Firmware independently switches off every UV channel on opening;
VIS remains usable. No wiring, firmware commands, settings schema, or operator
data are changed by this software integration.

## Serial and illumination behavior

`door_interlock.py` retains complete and fragmented reports from command replies
and serial drains. Malformed/inconsistent reports fail closed. Discovery consumes
a complete acknowledged identity, turns every output off, and obtains a valid
initial `M119` report before publishing the real serial handle. Reconnection uses
the current sensors rather than the previous handle's cache.

`TRIP` is compared by inequality, so a brief opening, wrap, or counter reset
invalidates active UV even after closure. UV ON has an authoritative door query
both before and after the command. The second query covers openings before ON
whose asynchronous reports have not yet arrived. A changed/unknown readback
causes explicit all-off and rejects activation. A failed all-off retains the
thermal deadline until physical OFF can be confirmed. Missing door adapters also
block UV. `Error:TS_DOOR_OPEN` remains a failure even when followed by `ok`.

Door polling uses the serial lock without waiting behind another owner and skips
commands while `motion_busy` is set. Poll failure does not skip thermal monitoring.
Unknown state does not erase an active UV deadline. The four-channel monitor stays
enabled during acquisition and every pause/retry.

`GET /api/lights/status` sends no hardware commands. Its repeatable response includes:

```json
{
  "door_interlock": {
    "known": false,
    "front_open": null,
    "top_open": null,
    "uv_blocked": true,
    "trip_count": null
  }
}
```

Manual door-blocked UV activation returns HTTP 409 and `{error, code: "E1402",
popup: true}`. Both error-message catalogs retain the same Hungarian message.
The top navigation bar displays separate yellow outlined icon-and-text badges for
open doors, with white text and icons. Known closed doors show no badge. Unknown
status remains visible while connected, and status is announced in Hungarian.
UV activation is disabled for open/unknown inputs; an existing
UV channel can still be turned off. Cached indicators follow firmware trips.
The badges read “Ajtó nyitva” and “Fedél nyitva”, with simplified enclosure SVGs
showing the open door or raised lid. Hovering or focusing a blocked UV control
explains in Hungarian which opening must be closed, or that its status is unknown.

## Automatic measurement

The start button requires known closed inputs even for a VIS-only plan.
`POST /api/auto_measurement/preflight` checks physical state before starting the
run or homing; direct first-tablet requests also reject open/unknown state before
folder creation or movement. The frontend waits for closure between homing axes.

`measurement_safety.py` owns cooperative pauses in the active tablet request.
It installs thread-local checkpoints for bounded motion and camera acquisitions;
preview and thermal-monitor threads remain independent. An accepted command or
camera retrieval can finish before the worker reaches its next checkpoint; firmware
UV cutoff does not wait for that checkpoint. While paused the worker turns lamps
off and waits without holding camera or serial locks. Each sensor query/wait is
finite and the existing autofocus abort endpoint cancels a paused request.

Closing both doors resumes automatically. Autofocus/presence phases restore their
optics and retry; planned captures restart the fresh guard/capture pair. A frame
whose exposure saw a trip is never saved. Previously saved rows stay saved and
are not repeated. Normal automatic capture now uses the same fresh acquisition
boundary as filter-series capture. Saving and live preview retain their existing
ownership boundaries.

Progress optionally includes `door_paused` and `door_pause_count`. The final
successful response also includes the count so a short pause cannot disappear
between progress polls. The Hungarian centered pause popup does not abort the
measurement and clears on resume. Brief completed trips receive a retained notice.
Stopping keeps scanner controls locked until the active backend request finishes
its cleanup. Stop includes the active `request_id` at `/api/abort-autofocus`;
the bounded cancellation cache also rejects a cancelled tablet POST that arrives
late, before resetting the abort flag or accessing devices. Cancellation records
expire after one hour and are capped at 128 entries. Incomplete tablet sets are
still excluded from inline analysis.

## Hardware-free verification (2026-10-15)

The focused backend suite passes all 88 tests; the five focused Angular spec
files pass all 12 tests. Angular application/template compilation passes. The
broader scanner run retains the three baseline failures described below; the
full suite and production build are not green.

Run focused backend checks from `backend/`:

```powershell
python -m unittest test_door_interlock test_measurement_safety test_light_control test_filter_series_camera test_virtual_octopus test_measurement_progress test_auto_measurement_settings test_inline_analysis
```

Focused Angular specs are `light-status.service.spec.ts`,
`door-interlock-status.component.spec.ts`, `door-safety.spec.ts`,
`motion-door-safety.spec.ts`, and `inline-measurement.spec.ts`. Run these with
`npm.cmd test -- --watch=false --browsers=ChromeHeadlessNoSandbox` and the appropriate
`--include=src/app/...` arguments. `npx.cmd ngc -p tsconfig.app.json --noEmit` verifies
application types and Angular templates independently of asset copying.

The local test build requires temporary Architect overrides that ignore the five
already unreadable legacy assets: `barlight_off.svg`, `ringlight_off.svg`,
`ringlight_on.svg`, `uv_ringlight_off.svg`, and `uv_ringlight_on.svg`. All other
assets, including the new door icons, are copied normally. The project build
configuration and the assets' permissions are unchanged. The default sandboxed
Chrome launcher fails its GPU process here; the built-in `ChromeHeadlessNoSandbox`
launcher works.

Production compilation retains the pre-existing initial-bundle and
pipeline-preview/step-inspector style budget failures, plus the existing PCA
template warning. A development build compiles but its unmodified asset-copy
step fails on the existing SVG access restrictions.

The broader backend scanner checks also retain three baseline failures in
`test_filter_revolver_api`: Y homing expects 173 instead of the current 163, and
two manual-autofocus tests receive HTTP 500. All three were reproduced against
the saved pre-task source in an isolated temporary directory.

Hardware acceptance still needs both Hall inputs, every UV channel and VIS,
brief opening during exposure, opening between activation checks, pause/closure
and stop during autofocus/capture/homing, USB reconnection, thermal cutoff, and
the packaged application. Packaging inputs are unchanged: Python modules use
static imports and SVGs use the existing Angular asset directory. No full Windows
package was built; existing production budgets prevent the normal package path.
