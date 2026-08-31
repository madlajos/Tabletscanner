import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import recipe_manager


class RecipeDirectoryTests(unittest.TestCase):
    def test_development_recipes_are_stored_next_to_module(self):
        with patch.object(sys, "frozen", False, create=True):
            expected = os.path.join(os.path.dirname(recipe_manager.__file__), "recipes")
            self.assertEqual(os.path.normpath(recipe_manager._recipes_dir()), os.path.normpath(expected))

    def test_frozen_recipes_are_stored_next_to_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = os.path.join(temp_dir, "app.exe")
            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "executable", executable),
            ):
                expected = os.path.join(temp_dir, "recipes")
                self.assertEqual(
                    os.path.normpath(recipe_manager._recipes_dir()),
                    os.path.normpath(expected),
                )
                self.assertTrue(os.path.isdir(expected))


if __name__ == "__main__":
    unittest.main()
