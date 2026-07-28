import unittest

from image_metadata import build_capture_metadata


class CaptureMetadataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
