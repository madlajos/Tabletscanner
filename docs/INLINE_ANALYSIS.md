# Inline recipe application

## Operator flow

1. Prepare and save a recipe in **Recept szerkesztő** as usual. Its first enabled node remains
   **Kép betöltése**. Select it under **Képelemző recept** on the scanner. The adjacent × turns
   analysis off for the next measurement.
2. Select tablets, a measurement name, and a capture plan. Start measurement. A private copy of
   the recipe is registered before imaging. Analysis setup or execution errors do not stop imaging.
3. Open **Recept alkalmazás** at any time. Its preview displays the last enabled recipe node and
   reuses the editor's ruler, pixel, scale, annotation and preview-save tools. The gallery retains
   the current measurement's tablet results; the preview pager selects additional output images.
   Histogram, intensity, particle, curve-fit and PCA outputs use the existing preview components.
4. **Elemzés bemenete** chooses one complete tablet or all completed tablets. The default is
   inferred from known cumulative nodes, including curve fitting and PCA; it can be overridden.
   **Összesítés csak a mérés végén** controls cumulative execution timing. With it off, the latest
   released set is recalculated after each tablet. With it on, cumulative execution waits for finish.
   Per-tablet recipes continue independently of this cumulative-only switch.
5. **Részletek** exposes supported final-node output settings (including intensity mean/median).
   Edits and **Újraszámítás** rerun the recorded inputs asynchronously. The saved recipe is unchanged.
   **Adatok letöltése** exports the selected result's serialized side outputs as JSON.
6. Analysis failures appear as dismissible local notifications. An unread failure makes the
   application tab red with an exclamation mark. Opening the tab acknowledges that indicator.

## Capture and input contract

Only a successfully completed tablet cycle supplies inputs. Original capture paths are selected
using the save callback's `masked` flag; partial sets and background-subtracted copies are excluded.
The next successful XY move and M400 completion release previously staged tablets. A timeout does
not release them. Finish releases the last complete tablet; stop/failure waits for any registered
capture request's cleanup. A late request for a cancelled session exits before device access.

The worker receives explicit `{path, tablet_id, metadata}` records. Every load-image node, including
secondary branches, receives that live set instead of its saved file path/order. Editor execution
without `live_inputs` retains its former behavior. No recipe schema migration is required.

Capture EXIF now includes `channel: "red" | "green" | "blue" | null`, translated from the configured
Piros/Zöld/Kék (or English equivalent) filter name. Unknown/empty/UV filters remain null. Illumination
continues to be recorded separately as `wavelength`, such as `uv255` or `vis`.

For **Pszeudokép képekből** and automatic RGB composition, each tablet/wavelength group has stable
source positions **1=Piros, 2=Zöld, 3=Kék**, independent of filenames, capture order, and filter slots.
Saved source selectors and channel transforms keep their meaning. For example, a monochrome RGB
composite uses red `1-GRAY`, green `2-GRAY`, blue `3-GRAY`. Non-RGB frames are available to other nodes
but are excluded from RGB trios. Missing/duplicate RGB captures fail the analysis with a clear error.
Focus stacking groups by tablet, illumination and optical filter before any downstream RGB assembly.
Reduced outputs retain source/identity metadata so cumulative gallery selection stays correct.

## Ownership, storage and limits

`inline_analysis.py` owns session state behind an RLock/condition and runs one daemon CPU worker.
It holds no camera/serial handles or hardware locks. HTTP commands only register work; status polling
and cached preview reads do not run a recipe. A long computation therefore has no analysis POST
timeout. Pending cumulative work is coalesced; stale results after settings changes are discarded.
Each execution has a distinct revision so the UI refreshes even when it misses intermediate states.

The service retains up to eight sessions, 256 queued jobs, and 1000 tablets per session. Decoded
input size is limited to 768 MiB per execution; oversized cumulative batches report a capacity error
and preserve the captures. This is an input guard, not a guarantee on intermediate recipe memory.
Large cumulative datasets may need smaller measurement groups. Processing uses full resolution;
images are not silently downscaled for analytical results.

Artifacts live in `<measurement folder>/elemzes/<session_id>/`: `recipe.json`, `session.json`, and
per-execution directories containing a recipe snapshot, a JSON data manifest and lossless NumPy
arrays. JSON publication is atomic; arrays are read with memory mapping and never pickle. Original
captures and existing operator recipes/settings are not overwritten. There is no automatic artifact
deletion. Current UI session state is in memory; reopening historical sessions after backend restart
is not implemented. Existing save-image/save-array nodes retain their preview-only runtime behavior;
inline results are persisted by the analysis service itself.

## API

All paths start with `/api/inline-analysis`. New API failures use `{error, code, popup}` and non-2xx
HTTP statuses. A successful status read can describe a failed background job. E4501–E4504 are shared
in both error-message catalogs; node-specific processing errors remain in the result details.

| Method/path | Request / result |
| --- | --- |
| `POST /sessions` | `{pipeline, measurement_name, measurement_folder, mode?, input_scope?}` → session, HTTP 201 |
| `GET /sessions/<id>` | Current session, tablet/result summaries, execution revisions, errors |
| `POST /sessions/<id>/finish` | `{cancelled?: boolean}` → finishing/completed state, HTTP 202 |
| `POST /sessions/<id>/configure` | `{pipeline?, mode?, input_scope?}` → queued recalculation, HTTP 202 |
| `GET /sessions/<id>/results/<result>` | `?image_index=0` → editor-compatible PreviewResponse |
| `GET /sessions/<id>/results/<result>/thumbnail` | `?image_index=0` → JPEG of that output page |

`POST /api/auto_measurement/step` accepts optional `inline_analysis_session_id`. Its hardware
acceptance, capture timing, clamping, thermal monitor and cleanup remain owned by the scanner.

## Validation

Hardware-free backend seams:

```powershell
cd backend
python -m pytest test_inline_analysis.py test_pipeline_live_inputs.py test_image_metadata.py test_pipeline_preview_side_outputs.py test_measurement_progress.py -q
$env:PYTHONUTF8 = '1'
python test_select_channel.py
```

The combined backend seam run passes 58 tests plus six metadata subtests. Extending that command
with `test_pseudo_image.py test_automatic_rgb_composite.py test_focus_stack.py
test_auto_measurement_settings.py` passes 105 tests plus six metadata subtests. No physical
camera/controller was used by tests.

Focused Angular checks (set `CHROME_BIN` to an installed Chrome/Chromium executable when needed):

```powershell
cd frontend
npm test -- --watch=false --browsers=ChromeHeadless --include=src/app/services/inline-analysis.service.spec.ts --include=src/app/services/pipeline-external-preview.spec.ts --include=src/app/features/auto-measurement/inline-measurement.spec.ts --include=src/app/components/analysis-recipe-selector/analysis-recipe-selector.component.spec.ts --include=src/app/app.component.spec.ts --include=src/app/features/recipe-creator/pca-chart.component.spec.ts
npm run build
```

All 18 focused specs pass. A full run after repairing the empty standalone app test had 76 passes
and six existing failures: MessageDisplay creation, MotionControl creation, ErrorPopupList creation,
MeasurementResultsPopup creation, XY collision settings, and filter-capture button setup. The new
PCA spec was added and passed afterward. Production Angular compiles but fails its existing budgets:
initial bundle approximately 1.35 MB versus 1 MB, preview CSS 19.98 kB versus 10 kB, inspector CSS
21.78 kB versus 10 kB. Before this feature the initial bundle was 1.31 MB; both style errors existed.

Browser integration (requires Python Playwright plus Chromium, and the development UI on port 4200):

```powershell
# Terminal 1, frontend/
npm start
# Terminal 2, backend/
python run_inline_browser_check.py
```

The browser check intercepts **every** backend request and uses a temporary Flask test client with
the real analysis worker. It never reaches the physical scanner or modifies operator data. It
selects A4/B4/C4, executes cropping, changes output settings, verifies cumulative tablet/page mapping,
runs PCA at measurement end, and checks that deliberate analysis failures preserve imaging and
produce the unread tab marker. Screenshots are written to `frontend/dist/inline-browser-check/`.

Physical acceptance remains: restart the source backend with these changes, use an operator-approved
capture plan on A4/B4/C4, verify next-position release and final flush, then test shuffled RGB plans,
missing channels, both cumulative modes, switching tabs during capture, and stop during a partial
tablet. The automated browser check above uses generated captures, not hardware.

No packaging inputs changed: modules are static Python imports, frontend assets use the existing
Angular path, and artifacts are written inside the already writable measurement folder. The full
Windows package was not rebuilt. The pre-existing root ignore rule hiding Electron source and the
packaged exit behavior remain documented limitations in README/AGENTS.
