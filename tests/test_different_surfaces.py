"""Test Suite for Different Surface Verification.

Verifies that:
1. Two different images of different surfaces produce 0 false correspondences and 0 matched craters.
2. The is_matched status is correctly set to False.
3. No fake fallback matches or Delaunay graphs are generated for unrelated scenes.
4. Overlapping images of the same surface are accurately matched and mapped.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import cv2
from app.pipeline.pds4_reader import LunarImageProduct, PDS4Metadata
from app.pipeline.manager import LunarPipelineManager
from test_matching import create_synthetic_lunar_scene

def test_different_surfaces_rejection():
    print("=== Testing Different Surface Rejection ===")
    np.random.seed(42)
    img1 = np.random.uniform(0.2, 0.8, (256, 256)).astype(np.float32)
    # Add a couple of craters at specific locations
    cv2.circle(img1, (60, 60), 20, 0.1, -1)
    cv2.circle(img1, (60, 60), 23, 0.7, 2)

    np.random.seed(999)
    img2 = np.random.uniform(0.1, 0.6, (256, 256)).astype(np.float32)
    # Add completely different craters at different locations
    cv2.circle(img2, (200, 200), 30, 0.1, -1)
    cv2.circle(img2, (200, 200), 33, 0.8, 2)

    prod1 = LunarImageProduct("Surface_Area_Alpha.xml", img1, img1, PDS4Metadata(lines=256, samples=256))
    prod2 = LunarImageProduct("Surface_Area_Beta.xml", img2, img2, PDS4Metadata(lines=256, samples=256))

    mgr = LunarPipelineManager()
    res1 = mgr.process_single_image(prod1)
    res2 = mgr.process_single_image(prod2)

    print(f"  Area Alpha - craters: {len(res1.craters)}, features: {len(res1.keypoints)}")
    print(f"  Area Beta  - craters: {len(res2.craters)}, features: {len(res2.keypoints)}")

    match = mgr.match_two_specific_images(res1, res2)
    print(f"  Matched nodes: {match.total_matched_features}")
    print(f"  Matched craters: {match.matched_craters_count}")
    print(f"  Edges: {len(match.edges)}")
    print(f"  Is matched: {match.is_matched}")
    print(f"  Status message: {match.status_message}")

    assert match.total_matched_features == 0, f"Expected 0 matched features, got {match.total_matched_features}"
    assert match.matched_craters_count == 0, f"Expected 0 matched craters, got {match.matched_craters_count}"
    assert len(match.edges) == 0, f"Expected 0 edges, got {len(match.edges)}"
    assert match.is_matched is False, "Expected is_matched to be False"
    print("Different surface rejection test PASSED!\n")

def test_same_surface_mapping():
    print("=== Testing Same Surface Alignment and Mapping ===")
    img1 = create_synthetic_lunar_scene(256, 256)
    h, w = img1.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), 3.0, 1.0)
    M[0, 2] += 10.0
    M[1, 2] += 6.0
    img2 = cv2.warpAffine(img1, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    prod1 = LunarImageProduct("LROC_FRAME_A.xml", img1, img1, PDS4Metadata(lines=h, samples=w))
    prod2 = LunarImageProduct("LROC_FRAME_B.xml", img2, img2, PDS4Metadata(lines=h, samples=w))

    mgr = LunarPipelineManager()
    res1 = mgr.process_single_image(prod1)
    res2 = mgr.process_single_image(prod2)

    match = mgr.match_two_specific_images(res1, res2)
    print(f"  Same surface matched nodes: {match.total_matched_features}")
    print(f"  Same surface matched craters: {match.matched_craters_count}")
    print(f"  Edges: {len(match.edges)}")
    print(f"  Is matched: {match.is_matched}")

    assert match.is_matched is True, "Expected is_matched to be True for overlapping frames"
    assert match.total_matched_features > 0, "Expected > 0 matched features"
    assert match.matched_craters_count > 0, "Expected > 0 matched craters"
    print("Same surface alignment and mapping test PASSED!\n")

if __name__ == "__main__":
    test_different_surfaces_rejection()
    test_same_surface_mapping()
    print("ALL TESTS IN test_different_surfaces.py PASSED SUCCESSFULLY!")
