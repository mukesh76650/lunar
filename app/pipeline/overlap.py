"""Overlap Determination Module.

Computes the geographic and pixel intersection footprint between the reference anchor
image and the warped target image.
"""
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2

@dataclass
class OverlapResult:
    overlap_mask: np.ndarray              # 2D uint8 mask (255 inside overlap, 0 outside)
    overlap_area_pixels: int              # Total intersection area in pixels
    overlap_ratio_anchor: float           # Percentage of anchor covered [0, 1]
    overlap_ratio_target: float           # Percentage of target contained in overlap [0, 1]
    bounding_box: Tuple[int, int, int, int] # (x, y, width, height)
    contour_points: List[List[int]]       # List of [x, y] polygon boundary vertices
    has_valid_overlap: bool               # True if overlap is non-trivial (> 2% of area)
    blended_preview: np.ndarray           # RGB visualization blending anchor & warped target

class OverlapAnalyzer:
    """Calculates overlap geometry and generates visual overlays."""

    @staticmethod
    def compute_overlap(
        anchor_img: np.ndarray,
        warped_target: np.ndarray,
        warped_mask: np.ndarray
    ) -> OverlapResult:
        """Compute the intersection between anchor frame and warped target footprint."""
        ha, wa = anchor_img.shape[:2]
        total_anchor_pixels = ha * wa

        # Anchor valid mask (non-zero or full frame)
        anchor_mask = (anchor_img > 0).astype(np.uint8) * 255
        if np.count_nonzero(anchor_mask) < total_anchor_pixels * 0.1:
            # If anchor has many 0s from normalization, treat entire bounding rect as valid
            anchor_mask = np.ones((ha, wa), dtype=np.uint8) * 255

        # Overlap mask is intersection
        overlap_mask = cv2.bitwise_and(anchor_mask, warped_mask)
        overlap_area = int(np.count_nonzero(overlap_mask))

        target_valid_pixels = int(np.count_nonzero(warped_mask))
        ratio_anchor = float(overlap_area / total_anchor_pixels) if total_anchor_pixels > 0 else 0.0
        ratio_target = float(overlap_area / target_valid_pixels) if target_valid_pixels > 0 else 0.0

        # Extract bounding box and polygon contour
        contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_vertices: List[List[int]] = []
        bbox = (0, 0, 0, 0)

        if contours:
            # Largest contour
            largest_c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_c)
            bbox = (int(x), int(y), int(w), int(h))

            # Approximate polygon for smooth JSON representation
            epsilon = 0.005 * cv2.arcLength(largest_c, True)
            approx = cv2.approxPolyDP(largest_c, epsilon, True)
            polygon_vertices = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

        has_valid_overlap = bool(ratio_anchor >= 0.02 and overlap_area > 100)

        # Create blended visualization (Anchor = Red/Green channel, Warped Target = Green/Blue channel)
        anchor_u8 = np.clip(anchor_img * 255, 0, 255).astype(np.uint8)
        warped_u8 = np.clip(warped_target * 255, 0, 255).astype(np.uint8)

        # False-color overlay: Cyan/Magenta or Green/Magenta difference
        preview = np.zeros((ha, wa, 3), dtype=np.uint8)
        preview[:, :, 0] = warped_u8        # Blue: Warped Target
        preview[:, :, 1] = anchor_u8        # Green: Anchor Reference
        preview[:, :, 2] = anchor_u8 // 2 + warped_u8 // 2  # Red blend

        # Highlight boundary in gold/yellow
        if contours:
            cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)

        return OverlapResult(
            overlap_mask=overlap_mask,
            overlap_area_pixels=overlap_area,
            overlap_ratio_anchor=round(ratio_anchor, 4),
            overlap_ratio_target=round(ratio_target, 4),
            bounding_box=bbox,
            contour_points=polygon_vertices,
            has_valid_overlap=has_valid_overlap,
            blended_preview=preview
        )
