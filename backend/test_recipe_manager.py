import os
import json
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


class RecipeFolderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.recipes_dir = self.temp_dir.name
        self.dir_patch = patch.object(recipe_manager, "_recipes_dir", return_value=self.recipes_dir)
        self.dir_patch.start()
        with open(os.path.join(self.recipes_dir, "Minta.json"), "w", encoding="utf-8") as file:
            json.dump({
                "schema_version": 1,
                "name": "Minta",
                "description": "Teszt",
                "steps": [],
                "created_at": "",
                "modified_at": "",
            }, file)

    def tearDown(self):
        self.dir_patch.stop()
        self.temp_dir.cleanup()

    def test_folder_crud_and_recipe_assignment_do_not_modify_recipe_document(self):
        recipe_path = os.path.join(self.recipes_dir, "Minta.json")
        with open(recipe_path, "r", encoding="utf-8") as file:
            original_recipe = file.read()

        folder, error = recipe_manager.create_recipe_folder("Vizsgálatok")
        self.assertIsNone(error)
        self.assertEqual(recipe_manager.list_recipe_folders(), [folder])

        success, error = recipe_manager.assign_recipe_folder("Minta", folder["id"])
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertEqual(recipe_manager.list_recipes()[0]["folder_id"], folder["id"])

        duplicate_name, error = recipe_manager.duplicate_recipe("Minta")
        self.assertIsNone(error)
        summaries = {recipe["name"]: recipe for recipe in recipe_manager.list_recipes()}
        self.assertEqual(summaries[duplicate_name]["folder_id"], folder["id"])

        renamed, error = recipe_manager.rename_recipe_folder(folder["id"], "Archivált")
        self.assertIsNone(error)
        self.assertEqual(renamed["name"], "Archivált")

        success, error = recipe_manager.delete_recipe_folder(folder["id"])
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(all(recipe["folder_id"] is None for recipe in recipe_manager.list_recipes()))
        with open(recipe_path, "r", encoding="utf-8") as file:
            self.assertEqual(file.read(), original_recipe)

    def test_folder_names_are_unique_ignoring_case(self):
        folder, error = recipe_manager.create_recipe_folder("Mérések")
        self.assertIsNotNone(folder)
        self.assertIsNone(error)
        duplicate, error = recipe_manager.create_recipe_folder("  mérések  ")
        self.assertIsNone(duplicate)
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
