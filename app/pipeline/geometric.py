"""RANSAC Geometric Verification and Transformation Estimation Module.

Performs robust model fitting (Homography / Affine) independently on candidate
correspondences produced by SuperPoint and LoFTR.
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from app.config import RANSACConfig

@dataclass
class GeometricVerificationResult:
    algorithm: str
    homography: Optional[np.ndarray]      # 3x3 Homography mapping target -> anchor
    inlier_mask: np.ndarray               # Boolean mask of inliers (length = num_matches)
    total_matches: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse: float
    anchor_inliers: np.ndarray            # (K, 2) coords
    target_inliers: np.ndarray            # (K, 2) coords
    anchor_outliers: np.ndarray           # (O, 2) coords
    target_outliers: np.ndarray           # (O, 2) coords
    is_valid: bool
    status_message: str

class GeometricVerifier:
    """Estimates geometric transformation using RANSAC and isolates inliers."""

    def __init__(self, config: Optional[RANSACConfig] = None):
        self.config = config or RANSACConfig()

    def verify(
        self,
        algorithm_name: str,
        pts_anchor: np.ndarray,
        pts_target: np.ndarray
    ) -> GeometricVerificationResult:
        """Estimate homography matrix H mapping pts_target -> pts_anchor using RANSAC.
        
        Args:
            algorithm_name: 'SuperPoint' or 'LoFTR'
            pts_anchor: (N, 2) array of correspondence coordinates in anchor frame
            pts_target: (N, 2) array of correspondence coordinates in target frame
        """
        n_matches = len(pts_anchor)
        if n_matches < 4:
            return GeometricVerificationResult(
                algorithm=algorithm_name,
                homography=None,
                inlier_mask=np.zeros((n_matches,), dtype=bool),
                total_matches=n_matches,
                inlier_count=0,
                inlier_ratio=0.0,
                reprojection_rmse=float('inf'),
                anchor_inliers=np.empty((0, 2), dtype=np.float32),
                target_inliers=np.empty((0, 2), dtype=np.float32),
                anchor_outliers=pts_anchor,
                target_outliers=pts_target,
                is_valid=False,
                status_message=f"Insufficient correspondences ({n_matches} < 4 required for RANSAC)."
            )

        # Estimate homography mapping target points to anchor points
        # pts_target -> pts_anchor
        try:
            H, mask = cv2.findHomography(
                pts_target,
                pts_anchor,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.reprojection_threshold,
                maxIters=self.config.max_iters,
                confidence=self.config.confidence
            )
        except Exception as e:
            H = None
            mask = None

        if H is None or mask is None:
            # Fallback: Affine partial 2D (4 DOF: rotation, scale, translation)
            try:
                affine_mat, mask = cv2.estimateAffinePartial2D(
                    pts_target,
                    pts_anchor,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.config.reprojection_threshold
                )
                if affine_mat is not None:
                    H = np.vstack([affine_mat, [0, 0, 1]])
            except Exception:
                H = None
                mask = None

        if H is None or mask is None:
            return GeometricVerificationResult(
                algorithm=algorithm_name,
                homography=None,
                inlier_mask=np.zeros((n_matches,), dtype=bool),
                total_matches=n_matches,
                inlier_count=0,
                inlier_ratio=0.0,
                reprojection_rmse=float('inf'),
                anchor_inliers=np.empty((0, 2), dtype=np.float32),
                target_inliers=np.empty((0, 2), dtype=np.float32),
                anchor_outliers=pts_anchor,
                target_outliers=pts_target,
                is_valid=False,
                status_message="RANSAC failed to converge on a valid geometric model."
            )

        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / n_matches) if n_matches > 0 else 0.0

        anchor_inliers = pts_anchor[inlier_mask]
        target_inliers = pts_target[inlier_mask]
        anchor_outliers = pts_anchor[~inlier_mask]
        target_outliers = pts_target[~inlier_mask]

        # Calculate exact reprojection RMSE for inliers
        if inlier_count > 0:
            pts_target_homo = np.hstack([target_inliers, np.ones((inlier_count, 1))])
            projected = (H @ pts_target_homo.T).T
            # Normalize by z coordinate
            projected_pts = projected[:, :2] / (projected[:, 2:3] + 1e-9)
            errors = np.linalg.norm(anchor_inliers - projected_pts, axis=1)
            rmse = float(np.sqrt(np.mean(errors ** 2)))
        else:
            rmse = float('inf')

        is_valid = inlier_count >= self.config.min_inliers_required and np.isfinite(rmse)

        return GeometricVerificationResult(
            algorithm=algorithm_name,
            homography=H,
            inlier_mask=inlier_mask,
            total_matches=n_matches,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_rmse=rmse,
            anchor_inliers=anchor_inliers,
            target_inliers=target_inliers,
            anchor_outliers=anchor_outliers,
            target_outliers=target_outliers,
            is_valid=is_valid,
            status_message="Valid transformation estimated." if is_valid else f"Low inlier count ({inlier_count} < {self.config.min_inliers_required})."
        )
