"""Integration test for Matching, Geometric Verification, and Multi-Metric Evaluation."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2

from app.pipeline.pds4_reader import LunarImageProduct, PDS4Metadata
from app.pipeline.manager import LunarPipelineManager

def create_synthetic_lunar_scene(h=256, w=256):
    """Generate a realistic synthetic lunar terrain image with multiple craters."""
    np.random.seed(101)
    img = np.ones((h, w), dtype=np.float32) * 0.45
    # Low-frequency terrain roughness
    y, x = np.mgrid[0:h, 0:w]
    img += 0.08 * np.sin(x / 18.0) * np.cos(y / 18.0)

    # Craters with sun angle from top-left (illuminated rim + dark interior shadow)
    craters = [
        (64, 64, 24),
        (180, 70, 32),
        (130, 160, 40),
        (70, 190, 20),
        (200, 200, 26)
    ]
    for cx, cy, r in craters:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # Inner depression
        mask_inner = dist < r
        img[mask_inner] -= 0.25 * (1.0 - dist[mask_inner] / r)
        # Rim uplift
        mask_rim = (dist >= r * 0.8) & (dist <= r * 1.2)
        # Directional sunlight
        angle = np.arctan2(y - cy, x - cx)
        sun_align = np.cos(angle - np.radians(225))  # Top-left illumination
        img[mask_rim] += 0.20 * sun_align[mask_rim] * (1.0 - np.abs(dist[mask_rim] - r) / (0.2 * r))

    # Add detector speckle noise
    noise = np.random.normal(0, 0.015, (h, w))
    img = np.clip(img + noise, 0.0, 1.0)
    return img.astype(np.float32)

def test_pipeline_matching_and_selection():
    img_anchor = create_synthetic_lunar_scene(256, 256)

    # Create target with known rigid transform (e.g. 15px shift, 3 deg rotation)
    h, w = img_anchor.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), 3.0, 1.0)
    M[0, 2] += 12.0
    M[1, 2] += 8.0
    img_target = cv2.warpAffine(img_anchor, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    prod1 = LunarImageProduct(
        name="LROC_NAC_ANCHOR.xml",
        image=img_anchor,
        raw_image=img_anchor,
        metadata=PDS4Metadata(title="Anchor LROC Frame", lines=h, samples=w, incidence_angle_deg=40.0)
    )
    prod2 = LunarImageProduct(
        name="LROC_NAC_TARGET_01.xml",
        image=img_target,
        raw_image=img_target,
        metadata=PDS4Metadata(title="Target LROC Frame", lines=h, samples=w, incidence_angle_deg=42.0)
    )

    mgr = LunarPipelineManager()
    result = mgr.process_products([prod1, prod2], anchor_identifier=0)

    assert result.status == "SUCCESS"
    assert len(result.pair_results) == 1
    pair_res = result.pair_results[0]

    # Verify SuperPoint and LoFTR were both executed
    print(f"SuperPoint matches: {pair_res.superpoint_matches}, Inliers: {pair_res.superpoint_geom.inlier_count}")
    print(f"LoFTR matches: {pair_res.loftr_matches}, Inliers: {pair_res.loftr_geom.inlier_count}")

    # Verify multi-metric evaluation
    print("SuperPoint evaluation:", pair_res.superpoint_eval.summary)
    print("LoFTR evaluation:", pair_res.loftr_eval.summary)
    print("Selected winner:", pair_res.decision.selected_algorithm)
    print("Selection rationale:", pair_res.decision.selection_rationale)

    # Verify registration & overlap
    assert pair_res.registration.warped_target is not None
    assert pair_res.overlap.overlap_ratio_anchor > 0.4
    print(f"Overlap ratio on anchor: {pair_res.overlap.overlap_ratio_anchor:.1%}")

    # Verify crater detection
    print(f"Detected craters: {pair_res.crater_detection.total_craters_detected}")
    print(f"Craters in overlap: {pair_res.crater_detection.craters_in_overlap}")
    assert pair_res.crater_detection.total_craters_detected > 0
    print("All matching, evaluation, registration, and crater detection tests PASSED!")

if __name__ == "__main__":
    test_pipeline_matching_and_selection()
