"""Exercise the real Angular UI and analysis worker with isolated fake captures.

Requires a development UI on localhost:4200 and Python Playwright/Chromium.
Every backend request is intercepted: this check never contacts real devices or
the running production backend, and all measurement files live in a temp folder.
Run manually; this is not a hardware acceptance test.
"""

from copy import deepcopy
import json
import logging
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, unquote, urlsplit

from flask import Flask
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright, expect

from inline_analysis import InlineAnalysisService
from inline_analysis_api import create_inline_analysis_blueprint
from pipeline_steps import STEP_DEFINITIONS


def step(name, **params):
    return {'instance_id': name, 'step_def_id': name, 'enabled': True,
            'param_values': {**{p.name: p.default for p in STEP_DEFINITIONS[name].params}, **params}}


def main():
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    service = InlineAnalysisService()
    fixture = Flask('inline-browser-fixture')
    fixture.register_blueprint(create_inline_analysis_blueprint(service))
    client = fixture.test_client()
    recipes = {
        'Inline crop test': {'schema_version': 1, 'name': 'Inline crop test', 'steps': [
            step('load_image', source='missing-editor-path'),
            step('mask_rect_roi', roi_x=20, roi_y=15, roi_width=100, roi_height=80, output_mode='crop'),
        ], 'connections': []},
        'Inline error test': {'schema_version': 1, 'name': 'Inline error test', 'steps': [
            step('load_image', source='missing-editor-path'), step('apply_threshold'),
        ], 'connections': []},
        'Inline PCA test': {'schema_version': 1, 'name': 'Inline PCA test', 'steps': [
            step('load_image', source='missing-editor-path'), step('select_channel', space='GRAY', channel='R'),
            step('calculate_histograms', bins=16), step('histogram_pca', max_components=2),
        ], 'connections': []},
    }
    settings = json.loads((Path(__file__).parent / 'settings.json').read_text(encoding='utf-8'))
    camera = {'ExposureTime': 20000, 'Gain': 0, 'Gamma': 1}
    ranges = {'ExposureTime': {'min': 100, 'max': 1000000, 'inc': 1},
              'Gain': {'min': 0, 'max': 24, 'inc': 0.1}}
    captures, requests, page_errors = [], [], []
    try:
        with tempfile.TemporaryDirectory(prefix='tabletscanner-inline-browser-') as folder:
            settings['auto_measurement_settings'] = {
                'save_location': folder, 'save_autofocus_image': True, 'check_tablet_presence': False,
                'capture_plan': [{'wavelength': 'vis', 'brightness': 'full', 'filter_position': 4,
                                  'exposure_time': 20000, 'gain': 0}],
            }

            def backend(route):
                request = route.request
                url = urlsplit(request.url)
                path, query = unquote(url.path), parse_qs(url.query)
                body = request.post_data_json if request.post_data else {}
                requests.append((request.method, path))
                data = {}
                if path.startswith('/api/inline-analysis/'):
                    response = client.open(path + ('?' + url.query if url.query else ''),
                                           method=request.method, json=body if request.post_data else None)
                    route.fulfill(status=response.status_code, content_type=response.content_type, body=response.data)
                    return
                if path == '/api/auto_measurement/step':
                    session_id, request_id = body['inline_analysis_session_id'], body['request_id']
                    service.begin_capture(session_id, request_id)
                    service.release_after_move(session_id)
                    index = body['tablet_index']
                    label = f'{chr(65 + (index - 1) % 10)}{(index - 1) // 10 + 1}'
                    capture = Path(folder) / f'{len(captures)}-{label}.jpg'
                    image = Image.new('RGB', (160, 120), '#334155')
                    draw = ImageDraw.Draw(image)
                    draw.ellipse((25, 20, 120, 105), fill=('#ef4444', '#22c55e', '#3b82f6')[index % 3])
                    metadata = {'wavelength': 'vis', 'channel': 'blue', 'filter_name': 'Kék', 'tray_position': label}
                    exif = Image.Exif()
                    exif[0x010E] = json.dumps(metadata, ensure_ascii=True)
                    image.save(capture, exif=exif)
                    service.capture_finished(session_id, request_id, index, [{'path': str(capture), 'metadata': metadata}])
                    captures.append(label)
                    data = {'status': 'success', 'tablet_index': index, 'saved_images': [str(capture)]}
                elif path == '/api/auto_measurement/progress':
                    data = {'request_id': query.get('request_id', [''])[0], 'status': 'pending',
                            'images': [], 'warnings': [], 'active_plan_row_index': None}
                elif path == '/api/health':
                    data = {'ready': True}
                elif path.startswith('/api/status/'):
                    data = {'connected': True, 'streaming': False, 'virtual': True}
                elif path == '/api/get-other-settings':
                    category = query.get('category', ['other_settings'])[0]
                    data = {category: settings.get(category, {})}
                elif path in ('/api/get-camera-settings', '/api/settings/camera/combinations'):
                    data = {'camera_params': camera, 'ranges': ranges,
                            'camera_combination_settings': settings.get('camera_combination_settings', {})}
                elif path == '/api/auto-measurement/camera-config':
                    data = {'connected': True, 'values': camera, 'ranges': ranges}
                elif path == '/api/settings/autofocus':
                    data = {'autofocus_settings': {'channel': 'vis', 'brightness': 'full', 'filter_position': 4}}
                elif path == '/api/settings/filter':
                    data = {'filter_settings': settings['filter_settings']}
                elif path == '/api/settings/motion/advanced':
                    data = {'advanced_motion_settings': settings['advanced_settings']}
                elif path == '/api/get_motion_platform_position':
                    data = {'x': 1.5, 'y': 2, 'z': 22, 'homed': True}
                elif path == '/api/lights/status':
                    data = {'active_channel': None, 'active_mode': None, 'auto_turned_off': [],
                            'channels': {name: False for name in ('vis', 'uv255', 'uv310', 'uv365')}}
                elif path == '/api/pipeline/step-catalog':
                    data = {'steps': [definition.to_dict() for definition in STEP_DEFINITIONS.values()]}
                elif path == '/api/recipes':
                    data = {'recipes': [{'name': name, 'description': 'Isolated browser check', 'step_count': 2} for name in recipes]}
                elif path.startswith('/api/recipes/'):
                    data = deepcopy(recipes[path.rsplit('/', 1)[-1]])
                elif path == '/api/recipe-folders':
                    data = {'folders': []}
                elif path in ('/api/check-file-exists', '/api/check-folder-exists'):
                    data = {'exists': True}
                elif path in ('/api/get_thumbnail', '/api/start-video-stream'):
                    if captures:
                        route.fulfill(content_type='image/jpeg', body=next(Path(folder).glob('*.jpg')).read_bytes())
                    else:
                        route.fulfill(status=204)
                    return
                elif path == '/api/image-metadata':
                    with Image.open(query['path'][0]) as image:
                        data = {'metadata': json.loads(image.getexif()[0x010E])}
                # Other scanner/settings requests receive inert fixture responses.
                route.fulfill(content_type='application/json', body=json.dumps(data))

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width': 1600, 'height': 1000})
                page.on('pageerror', lambda error: page_errors.append(str(error)))
                page.route('http://localhost:5000/**', backend)
                page.goto('http://localhost:4200', wait_until='domcontentloaded')
                page.wait_for_selector('app-auto-measurement')
                expect(page.locator('#saveLocation')).to_have_value(folder.replace('\\', '/'))

                def measure(recipe_name, measurement_name):
                    page.get_by_role('button', name='Szkenner', exact=True).click()
                    page.evaluate("""() => {
                        const scanner = ng.getComponent(document.querySelector('app-auto-measurement'));
                        scanner.clearSelection();
                    }""")
                    page.locator('#measurementName').fill(measurement_name)
                    page.get_by_role('button', name='Képelemző recept kiválasztása', exact=True).click()
                    page.locator('.recipe-card').filter(has_text=recipe_name).click()
                    page.locator('.tablet-dot').filter(has_text='A4').click()
                    page.locator('.tablet-dot').filter(has_text='C4').click()
                    try:
                        expect(page.locator('app-auto-measurement .start-button')).to_be_enabled(timeout=15000)
                    except AssertionError:
                        print('Start disabled:', page.locator('app-auto-measurement .start-button').inner_text())
                        print('Browser errors:', page_errors)
                        raise
                    page.locator('app-auto-measurement .start-button').click()
                    page.wait_for_function("""() => {
                        const analysis = ng.getComponent(document.querySelector('app-root')).inlineAnalysis;
                        return analysis.session?.state === 'completed';
                    }""", timeout=30000)

                measure('Inline crop test', 'A4 B4 C4 crop browser test')
                assert captures == ['A4', 'B4', 'C4'], captures
                page.get_by_role('button', name='Recept alkalmazás', exact=True).click()
                page.wait_for_selector('app-recipe-applier app-pipeline-preview img')
                page.wait_for_function("""() => {
                    const result = ng.getComponent(document.querySelector('app-root')).inlineAnalysis.preview.response;
                    return result?.image_width === 100 && result?.image_height === 80;
                }""")
                expect(page.locator('app-recipe-applier .image-toolbar')).to_be_visible()
                expect(page.get_by_role('button', name='Vonalzó (távolságmérés)', exact=True)).to_be_visible()
                assert page.locator('app-recipe-applier').inner_text().count('A4') >= 1
                output = Path(__file__).parent.parent / 'frontend' / 'dist' / 'inline-browser-check'
                output.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(output / 'crop-results.png'))
                page.locator('#inline-param-output_mode').select_option('mask')
                page.wait_for_function("""() =>
                    ng.getComponent(document.querySelector('app-root')).inlineAnalysis.preview.response?.image_width === 160""")
                page.locator('#inline-input-scope').select_option('cumulative')
                page.wait_for_function("""() => {
                    const analysis = ng.getComponent(document.querySelector('app-root')).inlineAnalysis;
                    return analysis.session?.input_scope === 'cumulative' && analysis.preview.response?.image_count === 3;
                }""")
                page.locator('.gallery-card').filter(has_text='B4').click()
                page.wait_for_function("""() =>
                    ng.getComponent(document.querySelector('app-root')).inlineAnalysis.preview.image_index === 1""")
                expect(page.locator('.gallery-card:not(.cumulative).selected')).to_have_count(1)
                page.get_by_role('switch').check()
                page.wait_for_function("""() =>
                    ng.getComponent(document.querySelector('app-root')).inlineAnalysis.session?.mode === 'measurement_end'""")
                measure('Inline PCA test', 'A4 B4 C4 cumulative PCA browser test')
                page.get_by_role('button', name='Recept alkalmazás', exact=True).click()
                expect(page.locator('app-recipe-applier app-pca-chart')).to_be_visible()
                page.screenshot(path=str(output / 'cumulative-pca.png'))
                page.get_by_role('switch').uncheck()
                page.locator('#inline-input-scope').select_option('tablet')
                page.wait_for_function("""() =>
                    ng.getComponent(document.querySelector('app-root')).inlineAnalysis.session?.input_scope === 'tablet'""")
                measure('Inline error test', 'A4 B4 C4 error browser test')
                assert captures == ['A4', 'B4', 'C4'] * 3, captures
                expect(page.locator('.nav-tab.analysis-error')).to_be_visible()
                page.locator('.nav-tab.analysis-error').click()
                expect(page.locator('.nav-tab.analysis-error')).to_have_count(0)
                page.wait_for_function("""() =>
                    ng.getComponent(document.querySelector('app-root')).inlineAnalysis.session.errors.length === 3""")
                page.screenshot(path=str(output / 'analysis-errors.png'))
                assert not page_errors, page_errors
                assert not any(path == '/api/pipeline/preview' for _, path in requests), 'Unexpected synchronous preview call'
                browser.close()
                print('PASS: A4/B4/C4 crop, output edits, cumulative PCA/mode toggle, per-tablet gallery, failure isolation and unread badge; no physical hardware used.')
                print(f'Screenshots: {output}')
    finally:
        service.close()
        if service._worker:
            service._worker.join(timeout=10)


if __name__ == '__main__':
    main()
