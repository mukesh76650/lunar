r"""Test Suite for Moon Dataset (D:\moon) and Training Pipeline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.pipeline.crater_trainer import CraterTrainingManager
from app.main import get_moon_training_dataset

class TestMoonDatasetIntegration(unittest.TestCase):

    def test_01_dataset_directory_structure(self):
        root = "D:/moon/moon"
        self.assertTrue(os.path.isdir(root), f"Expected dataset root {root} to exist")
        self.assertTrue(os.path.isdir(os.path.join(root, "images", "train")))
        self.assertTrue(os.path.isdir(os.path.join(root, "labels", "train")))
        self.assertTrue(os.path.isdir(os.path.join(root, "images", "val")))
        self.assertTrue(os.path.isdir(os.path.join(root, "labels", "val")))

    def test_02_yaml_configuration(self):
        yaml_path = "lunar_craters.yaml"
        self.assertTrue(os.path.isfile(yaml_path), f"Expected {yaml_path} to exist")
        with open(yaml_path, "r") as f:
            content = f.read()
            self.assertIn("D:/moon/moon", content)
            self.assertIn("crater", content)

    def test_03_moon_sample_endpoint(self):
        trainer = CraterTrainingManager()
        data = trainer.sample_from_moon_dataset("D:/moon/moon")
        self.assertIn("angles", data)
        self.assertGreaterEqual(len(data["angles"]), 2)
        self.assertIn("negative", data)
        self.assertEqual(data["angles"][0]["image"].shape, (64, 64))
        print(f"  Successfully loaded {len(data['angles'])} angles from D:/moon/moon: {[a['label'] for a in data['angles']]}")

    def test_04_api_moon_data_response(self):
        import json
        res = get_moon_training_dataset()
        data = json.loads(res.body.decode("utf-8"))
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["dataset_source"], "D:/moon/moon")
        self.assertGreaterEqual(len(data["angles"]), 2)

if __name__ == "__main__":
    unittest.main()
