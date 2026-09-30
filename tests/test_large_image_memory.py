"""Stress test for large lunar image memory management."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
import time

from app.pipeline.pds4_reader import LunarImageProduct, PDS4Metadata
from app.pipeline.manager import LunarPipelineManager

def test_large_image_pipeline():
    print("Testing pipeline on high-resolution lunar images (1600 x 1600)...")
    h, w = 1600, 1600
    
    # Create large synthetic lunar surface
    y, x = np.mgrid[0:h, 0:w]
    base = 0.45 + 0.05 * np.sin(x / 60.0) * np.cos(y / 60.0)
    # Add a few craters
    craters = [(400, 400, 80), (1200, 500, 120), (800, 1100, 100)]
    for cx, cy, r in craters:
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        inner = d < r
        base[inner] -= 0.2 * (1.0 - d[inner] / r)
        rim = (d >= r * 0.8) & (d <= r * 1.2)
        base[rim] += 0.15

    img_a = np.clip(base, 0.0, 1.0).astype(np.float32)

    # Shift target image
    M = np.float32([[1, 0, 35], [0, 1, -25]])
    img_b = cv2.warpAffine(img_a, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    p1 = LunarImageProduct("LARGE_ANCHOR.xml", img_a, img_a, PDS4Metadata("Large Anchor", lines=h, samples=w))
    p2 = LunarImageProduct("LARGE_TARGET.xml", img_b, img_b, PDS4Metadata("Large Target", lines=h, samples=w))

    mgr = LunarPipelineManager()
    t0 = time.time()
    res = mgr.process_products([p1, p2], anchor_identifier=0)
    t1 = time.time()

    print(f"High-res test completed in {t1 - t0:.2f}s without memory error!")
    assert res.status == "SUCCESS"
    pair = res.pair_results[0]
    print(f"Winner: {pair.decision.selected_algorithm}")
    print(f"LoFTR matches: {pair.loftr_matches}, inliers: {pair.loftr_geom.inlier_count}")
    print(f"SuperPoint matches: {pair.superpoint_matches}, inliers: {pair.superpoint_geom.inlier_count}")
    print(f"Overlap: {pair.overlap.overlap_ratio_anchor:.1%}")
    print(f"Detected craters: {pair.crater_detection.total_craters_detected}")

if __name__ == "__main__":
    test_large_image_pipeline()
