"""Multi-Metric Quantitative Evaluation Module for Registration Algorithms.

Evaluates SuperPoint and LoFTR results across:
1. Total Match Count
2. Inlier Count
3. Inlier Ratio
4. Reprojection RMSE
5. Spatial Distribution and Uniformity (Convex Hull Ratio + Grid Entropy)
6. Geometric Plausibility (Determinant, Condition Number, Corner Bounds)
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from app.config import EvaluationWeights
from app.pipeline.geometric import GeometricVerificationResult

@dataclass
class SpatialUniformityMetrics:
    convex_hull_area: float
    convex_hull_ratio: float           # Hull area / Image total area
    grid_entropy: float                # Normalized Shannon entropy across 4x4 spatial quadrants [0, 1]
    active_quadrants: int              # Number of non-empty grid cells (out of 16)
    spatial_uniformity_score: float    # Combined spatial score [0, 1]

@dataclass
class GeometricPlausibilityMetrics:
    is_invertible: bool
    determinant_2x2: float
    condition_number: float
    corners_convex: bool
    corners_within_bounds: bool
    is_plausible: bool
    plausibility_penalty: float

@dataclass
class AlgorithmEvaluationReport:
    algorithm: str
    total_matches: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse: float
    spatial: SpatialUniformityMetrics
    plausibility: GeometricPlausibilityMetrics
    composite_reliability_score: float
    is_reliable: bool
    summary: str

class RegistrationEvaluator:
    """Evaluates correspondence pipelines using a multi-criteria reliability framework."""

    def __init__(self, weights: Optional[EvaluationWeights] = None):
        self.weights = weights or EvaluationWeights()

    def compute_spatial_uniformity(
        self,
        inliers: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> SpatialUniformityMetrics:
        """Calculate spatial distribution and uniformity of inlier keypoints."""
        h, w = image_shape
        total_area = float(h * w)

        if len(inliers) < 3:
            return SpatialUniformityMetrics(
                convex_hull_area=0.0,
                convex_hull_ratio=0.0,
                grid_entropy=0.0,
                active_quadrants=0,
                spatial_uniformity_score=0.0
            )

        # 1. Convex Hull Area Ratio using OpenCV
        try:
            pts_f32 = inliers.reshape(-1, 1, 2).astype(np.float32)
            hull = cv2.convexHull(pts_f32)
            hull_area = float(cv2.contourArea(hull))
            hull_ratio = float(np.clip(hull_area / total_area, 0.0, 1.0))
        except Exception:
            hull_area = 0.0
            hull_ratio = 0.0

        # 2. Grid Dispersion Entropy (4x4 spatial partition)
        grid_rows, grid_cols = 4, 4
        num_cells = grid_rows * grid_cols
        x_bins = np.linspace(0, w, grid_cols + 1)
        y_bins = np.linspace(0, h, grid_rows + 1)

        counts, _, _ = np.histogram2d(inliers[:, 0], inliers[:, 1], bins=[x_bins, y_bins])
        counts = counts.flatten()
        total_pts = len(inliers)

        probs = counts / total_pts
        active_cells = int(np.sum(counts > 0))

        # Normalized Shannon entropy
        non_zero_p = probs[probs > 0]
        if len(non_zero_p) > 1:
            raw_entropy = -np.sum(non_zero_p * np.log2(non_zero_p))
            max_entropy = np.log2(num_cells)
            grid_entropy = float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))
        else:
            grid_entropy = 0.0

        # Blended spatial score
        spatial_score = float(0.5 * hull_ratio + 0.5 * grid_entropy)

        return SpatialUniformityMetrics(
            convex_hull_area=round(hull_area, 2),
            convex_hull_ratio=round(hull_ratio, 4),
            grid_entropy=round(grid_entropy, 4),
            active_quadrants=active_cells,
            spatial_uniformity_score=round(spatial_score, 4)
        )

    def compute_geometric_plausibility(
        self,
        homography: Optional[np.ndarray],
        target_shape: Tuple[int, int],
        anchor_shape: Tuple[int, int]
    ) -> GeometricPlausibilityMetrics:
        """Verify the physical plausibility and numerical stability of estimated transformation."""
        if homography is None:
            return GeometricPlausibilityMetrics(
                is_invertible=False,
                determinant_2x2=0.0,
                condition_number=float('inf'),
                corners_convex=False,
                corners_within_bounds=False,
                is_plausible=False,
                plausibility_penalty=1.0
            )

        H = homography

        # 1. Check invertibility and condition number via SVD
        try:
            _, s, _ = np.linalg.svd(H)
            cond = float(s[0] / (s[-1] + 1e-9))
            is_invertible = bool(s[-1] > 1e-7 and cond < 1e5)
        except Exception:
            cond = float('inf')
            is_invertible = False

        # 2. Check 2x2 determinant for reflection / folding
        det2x2 = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
        positive_det = det2x2 > 0.05  # Lunar imaging pairs maintain consistent rotational orientation

        # 3. Corner projection check
        ht, wt = target_shape
        ha, wa = anchor_shape
        corners = np.array([
            [0, 0, 1],
            [wt, 0, 1],
            [wt, ht, 1],
            [0, ht, 1]
        ], dtype=np.float32)

        proj_corners = (H @ corners.T).T
        proj_pts = proj_corners[:, :2] / (proj_corners[:, 2:3] + 1e-9)

        # Check if projected quad is convex using cross-product signs
        edges = np.roll(proj_pts, -1, axis=0) - proj_pts
        cross_z = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        corners_convex = bool(np.all(cross_z > 0) or np.all(cross_z < 0))

        # Check bounds: projected corners shouldn't expand uncontrollably (e.g. beyond 4x image size)
        x_min, y_min = proj_pts.min(axis=0)
        x_max, y_max = proj_pts.max(axis=0)
        within_bounds = bool(
            x_min > -3 * wa and x_max < 4 * wa and
            y_min > -3 * ha and y_max < 4 * ha
        )

        is_plausible = is_invertible and positive_det and corners_convex and within_bounds and (cond < 2000.0)

        # Penalty score
        penalty = 0.0
        if not positive_det:
            penalty += 0.5
        if not corners_convex:
            penalty += 0.4
        if not within_bounds:
            penalty += 0.3
        if cond > 2000.0:
            penalty += 0.3

        return GeometricPlausibilityMetrics(
            is_invertible=is_invertible,
            determinant_2x2=round(det2x2, 4),
            condition_number=round(cond, 2),
            corners_convex=corners_convex,
            corners_within_bounds=within_bounds,
            is_plausible=is_plausible,
            plausibility_penalty=float(min(1.0, penalty))
        )

    def evaluate(
        self,
        geom_result: GeometricVerificationResult,
        target_shape: Tuple[int, int],
        anchor_shape: Tuple[int, int]
    ) -> AlgorithmEvaluationReport:
        """Run comprehensive multi-metric evaluation on an algorithm's registration result."""
        spatial = self.compute_spatial_uniformity(geom_result.anchor_inliers, anchor_shape)
        plausibility = self.compute_geometric_plausibility(
            geom_result.homography,
            target_shape,
            anchor_shape
        )

        # Calculate composite reliability score
        # Normalization terms
        norm_inlier_count = float(np.clip(geom_result.inlier_count / 100.0, 0.0, 1.0))
        norm_inlier_ratio = float(geom_result.inlier_ratio)
        norm_spatial = float(spatial.spatial_uniformity_score)
        
        # RMSE penalty term: 0 error = 1.0, max_acceptable_rmse = 0.0
        max_rmse = self.weights.max_acceptable_rmse
        rmse = geom_result.reprojection_rmse
        if np.isfinite(rmse):
            norm_rmse_score = float(np.clip(1.0 - (rmse / max_rmse), 0.0, 1.0))
        else:
            norm_rmse_score = 0.0

        # Weighted composite score
        w = self.weights
        composite_score = (
            w.w_inlier_ratio * norm_inlier_ratio +
            w.w_inlier_count * norm_inlier_count +
            w.w_spatial_coverage * norm_spatial +
            w.w_rmse_penalty * norm_rmse_score
        )
        # Apply plausibility penalty
        composite_score = max(0.0, composite_score - plausibility.plausibility_penalty)

        # Reliability decision criteria
        is_reliable = bool(
            geom_result.is_valid and
            plausibility.is_plausible and
            geom_result.inlier_count >= w.min_acceptable_inliers and
            geom_result.reprojection_rmse <= w.max_acceptable_rmse and
            spatial.convex_hull_ratio >= w.min_acceptable_hull_ratio and
            composite_score > 0.25
        )

        summary = (
            f"Algorithm: {geom_result.algorithm} | "
            f"Matches: {geom_result.total_matches}, Inliers: {geom_result.inlier_count} ({geom_result.inlier_ratio:.1%}), "
            f"RMSE: {geom_result.reprojection_rmse:.2f}px, Hull Ratio: {spatial.convex_hull_ratio:.1%}, "
            f"Plausible: {plausibility.is_plausible}, Score: {composite_score:.3f}"
        )

        return AlgorithmEvaluationReport(
            algorithm=geom_result.algorithm,
            total_matches=geom_result.total_matches,
            inlier_count=geom_result.inlier_count,
            inlier_ratio=round(geom_result.inlier_ratio, 4),
            reprojection_rmse=round(geom_result.reprojection_rmse, 3),
            spatial=spatial,
            plausibility=plausibility,
            composite_reliability_score=round(composite_score, 4),
            is_reliable=is_reliable,
            summary=summary
        )
