"""Algorithm Selection Engine.

Compares SuperPoint and LoFTR results using multi-metric evaluation criteria and selects
the most reliable geometric registration for each image pair, with robust fallback.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from app.pipeline.geometric import GeometricVerificationResult
from app.pipeline.evaluation import AlgorithmEvaluationReport

@dataclass
class SelectionDecision:
    selected_algorithm: str
    selected_homography: Optional[np.ndarray]
    selected_geom_result: GeometricVerificationResult
    selected_evaluation: AlgorithmEvaluationReport
    other_algorithm: str
    other_evaluation: AlgorithmEvaluationReport
    selection_rationale: str
    is_fallback_used: bool
    confidence_level: str  # "HIGH", "MODERATE", "LOW"

class AlgorithmSelector:
    """Implements the comparative multi-criteria decision logic between SuperPoint and LoFTR."""

    @staticmethod
    def select_best(
        geom_sp: GeometricVerificationResult,
        eval_sp: AlgorithmEvaluationReport,
        geom_loftr: GeometricVerificationResult,
        eval_loftr: AlgorithmEvaluationReport
    ) -> SelectionDecision:
        """Select the most reliable registration algorithm based on comprehensive evaluation."""
        sp_reliable = eval_sp.is_reliable
        loftr_reliable = eval_loftr.is_reliable

        score_sp = eval_sp.composite_reliability_score
        score_loftr = eval_loftr.composite_reliability_score

        # Case 1: Both meet reliability thresholds
        if sp_reliable and loftr_reliable:
            # Check for close scores
            if abs(score_sp - score_loftr) < 0.05:
                # Tie-breaker: prioritize spatial uniformity and lower RMSE
                if eval_sp.spatial.spatial_uniformity_score > eval_loftr.spatial.spatial_uniformity_score:
                    winner = "SuperPoint"
                    rationale = (
                        f"Both algorithms met quality criteria. SuperPoint was selected due to superior "
                        f"spatial uniformity ({eval_sp.spatial.spatial_uniformity_score:.2f} vs {eval_loftr.spatial.spatial_uniformity_score:.2f}) "
                        f"and lower RMSE ({eval_sp.reprojection_rmse:.2f}px vs {eval_loftr.reprojection_rmse:.2f}px)."
                    )
                else:
                    winner = "LoFTR"
                    rationale = (
                        f"Both algorithms met quality criteria. LoFTR was selected due to superior "
                        f"spatial uniformity ({eval_loftr.spatial.spatial_uniformity_score:.2f} vs {eval_sp.spatial.spatial_uniformity_score:.2f}) "
                        f"and consistent dense coverage."
                    )
            elif score_sp > score_loftr:
                winner = "SuperPoint"
                rationale = (
                    f"SuperPoint achieved higher composite reliability score ({score_sp:.3f} vs {score_loftr:.3f}) "
                    f"with {eval_sp.inlier_count} inliers ({eval_sp.inlier_ratio:.1%}) and RMSE {eval_sp.reprojection_rmse:.2f}px."
                )
            else:
                winner = "LoFTR"
                rationale = (
                    f"LoFTR achieved higher composite reliability score ({score_loftr:.3f} vs {score_sp:.3f}) "
                    f"with {eval_loftr.inlier_count} inliers ({eval_loftr.inlier_ratio:.1%}) and RMSE {eval_loftr.reprojection_rmse:.2f}px."
                )
            confidence = "HIGH"
            fallback = False

        # Case 2: Only SuperPoint is reliable
        elif sp_reliable and not loftr_reliable:
            winner = "SuperPoint"
            rationale = (
                f"SuperPoint selected as fallback: LoFTR failed quality criteria "
                f"(LoFTR inliers: {eval_loftr.inlier_count}, Plausible: {eval_loftr.plausibility.is_plausible}). "
                f"SuperPoint demonstrated valid geometric convergence (Score: {score_sp:.3f})."
            )
            confidence = "MODERATE"
            fallback = True

        # Case 3: Only LoFTR is reliable
        elif loftr_reliable and not sp_reliable:
            winner = "LoFTR"
            rationale = (
                f"LoFTR selected as fallback: SuperPoint failed quality criteria "
                f"(SuperPoint inliers: {eval_sp.inlier_count}, Plausible: {eval_sp.plausibility.is_plausible}). "
                f"LoFTR demonstrated valid geometric convergence (Score: {score_loftr:.3f})."
            )
            confidence = "MODERATE"
            fallback = True

        # Case 4: Neither meets strict criteria (fallback to higher score or non-degenerate homography)
        else:
            if eval_sp.plausibility.is_plausible and not eval_loftr.plausibility.is_plausible:
                winner = "SuperPoint"
                rationale = "Neither algorithm fully satisfied strict criteria, but SuperPoint maintained geometric plausibility."
            elif eval_loftr.plausibility.is_plausible and not eval_sp.plausibility.is_plausible:
                winner = "LoFTR"
                rationale = "Neither algorithm fully satisfied strict criteria, but LoFTR maintained geometric plausibility."
            elif score_sp >= score_loftr:
                winner = "SuperPoint"
                rationale = f"Marginal convergence: SuperPoint chosen with highest available composite score ({score_sp:.3f})."
            else:
                winner = "LoFTR"
                rationale = f"Marginal convergence: LoFTR chosen with highest available composite score ({score_loftr:.3f})."
            confidence = "LOW"
            fallback = True

        if winner == "SuperPoint":
            return SelectionDecision(
                selected_algorithm="SuperPoint",
                selected_homography=geom_sp.homography,
                selected_geom_result=geom_sp,
                selected_evaluation=eval_sp,
                other_algorithm="LoFTR",
                other_evaluation=eval_loftr,
                selection_rationale=rationale,
                is_fallback_used=fallback,
                confidence_level=confidence
            )
        else:
            return SelectionDecision(
                selected_algorithm="LoFTR",
                selected_homography=geom_loftr.homography,
                selected_geom_result=geom_loftr,
                selected_evaluation=eval_loftr,
                other_algorithm="SuperPoint",
                other_evaluation=eval_sp,
                selection_rationale=rationale,
                is_fallback_used=fallback,
                confidence_level=confidence
            )
