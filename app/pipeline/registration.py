"""Image Registration Module.

Warps the target image into the anchor image coordinate system using the selected
reliable geometric transformation matrix H.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

@dataclass
class RegistrationResult:
    warped_target: np.ndarray             # Warped target image in anchor coordinates [0, 1]
    warped_mask: np.ndarray               # Binary mask (uint8) of valid warped pixels (255 where valid)
    anchor_shape: Tuple[int, int]         # (height, width) of anchor frame
    homography: np.ndarray                # 3x3 transformation matrix applied

class ImageRegistrar:
    """Performs perspective warping of lunar imagery into reference coordinates."""

    @staticmethod
    def warp_target_to_anchor(
        target_image: np.ndarray,
        homography: np.ndarray,
        anchor_shape: Tuple[int, int]
    ) -> RegistrationResult:
        """Warp target image to align with anchor coordinates using homography H.
        
        Args:
            target_image: Preprocessed 2D float32 target image
            homography: 3x3 matrix mapping target -> anchor coordinates
            anchor_shape: (height, width) of anchor image
        """
        ha, wa = anchor_shape
        ht, wt = target_image.shape

        # Warp image
        warped_img = cv2.warpPerspective(
            target_image,
            homography,
            (wa, ha),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )

        # Warp binary mask to trace exact valid footprint
        target_ones = np.ones((ht, wt), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            target_ones,
            homography,
            (wa, ha),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return RegistrationResult(
            warped_target=warped_img.astype(np.float32),
            warped_mask=warped_mask,
            anchor_shape=anchor_shape,
            homography=homography
        )
