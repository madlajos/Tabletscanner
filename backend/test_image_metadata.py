import json
import os
import tempfile
import unittest

from PIL import Image

from image_metadata import build_capture_metadata, serialize_capture_metadata, tray_position_label


class CaptureMetadataTests(unittest.TestCase):
    def test_hungarian_metadata_round_trips_through_jpeg_exif(self):
        metadata = {'filter_name': 'Kék áteresztő szűrő'}
        exif = Image.Exif()
        exif[0x010E] = serialize_capture_metadata(metadata)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'metadata.jpg')
            Image.new('RGB', (1, 1)).save(path, format='JPEG', exif=exif)
            with Image.open(path) as image:
                restored = json.loads(image.getexif()[0x010E])

        self.assertEqual(metadata, restored)

    def test_grid_labels_use_actual_coordinates_and_configured_geometry(self):
        settings = {'advanced_settings': {'first_tablet_x_mm': 3, 'first_tablet_y_mm': 2, 'tablet_spacing_mm': 15}}
        for position, expected in [({'x': 3, 'y': 2}, 'A1'), ({'x': 18, 'y': 2}, 'B1'),
                                   ({'x': 3, 'y': 17}, 'A2'), ({'x': 3.1, 'y': 2}, None),
                                   ({'x': 4, 'y': 2}, None), ({'x': 153, 'y': 2}, None),
                                   ({'x': None, 'y': 2}, None)]:
            self.assertEqual(expected, tray_position_label(settings, position))

    def test_capture_errors_are_preserved_as_a_list(self):
        metadata = build_capture_metadata(settings={}, position={}, wavelength='uv255',
            filter_position=1, camera_values={}, errors=['ZOffset difference: -1.5 mm'])
        self.assertEqual(['ZOffset difference: -1.5 mm'], metadata['Errors'])

    def test_builds_complete_metadata_from_runtime_state(self):
        settings = {
            "other_settings": {
                "objective": "1.0x",
                "spacer_rings": "1",
                "camera_settings_file": r"C:\profiles\scanner.pfs",
            },
            "filter_settings": {
                "filters": [
                    {
                        "id": "green",
                        "name": "Green",
                        "wavelength_range": "500-550 nm",
                    }
                ],
                "slots": [None, "green", None, None, None, None],
            },
        }

        metadata = build_capture_metadata(
            settings=settings,
            position={"x": 12.5, "y": 23.5, "z": 4.25},
            wavelength="uv365",
            filter_position=2,
            camera_values={
                "exposure_time": 75000.0,
                "gain": 2.5,
                "gamma": 1.2,
            },
        )

        self.assertEqual(metadata["x"], 12.5)
        self.assertEqual(metadata["y"], 23.5)
        self.assertEqual(metadata["z"], 4.25)
        self.assertEqual(metadata["wavelength"], "uv365")
        self.assertEqual(metadata["filter_position"], 2)
        self.assertEqual(metadata["filter_wavelength"], "500-550 nm")
        self.assertEqual(metadata["filter_name"], "Green")
        self.assertEqual(metadata["exposure_time"], 75000.0)
        self.assertEqual(metadata["gain"], 2.5)
        self.assertEqual(metadata["gamma"], 1.2)
        self.assertEqual(metadata["camera_profile"], r"C:\profiles\scanner.pfs")
        self.assertEqual(
            metadata["camera_settings_file"], r"C:\profiles\scanner.pfs"
        )

    def test_empty_filter_slot_is_recorded_without_inventing_a_filter(self):
        metadata = build_capture_metadata(
            settings={"filter_settings": {"filters": [], "slots": [None] * 6}},
            position={},
            wavelength="vis",
            filter_position=1,
            camera_values={},
        )

        self.assertEqual(metadata["filter_position"], 1)
        self.assertIsNone(metadata["filter_name"])
        self.assertIsNone(metadata["filter_wavelength"])
        self.assertIsNone(metadata["x"])
        self.assertIsNone(metadata["exposure_time"])

    def test_live_camera_values_override_stale_client_metadata(self):
        metadata = build_capture_metadata(
            settings={},
            position={},
            wavelength="vis",
            filter_position=None,
            camera_values={"gain": 4.0},
            requested_metadata={"gain": 99.0, "gamma": 1.1},
        )

        self.assertEqual(metadata["gain"], 4.0)
        self.assertEqual(metadata["gamma"], 1.1)

    def test_floating_point_noise_is_rounded_for_json_and_exif(self):
        metadata = build_capture_metadata(
            settings={}, position={'x': 0.0, 'y': 21.80000000000001, 'z': 16.037611065434337},
            wavelength='uv255', filter_position=2,
            camera_values={'exposure_time': 1000000.0, 'gain': 9.999998795594486, 'gamma': 1.00000001})
        self.assertEqual(0, metadata['x'])
        self.assertEqual(21.8, metadata['y'])
        self.assertEqual(16.0376, metadata['z'])
        self.assertEqual(1000000, metadata['exposure_time'])
        self.assertEqual(10, metadata['gain'])
        self.assertEqual(1, metadata['gamma'])


if __name__ == "__main__":
    unittest.main()
