"""Pipeline Manager Module.

Provides:
1. Single-image independent processing: features & craters extraction.
2. Sequential processing across n images with cached results.
3. Two-image feature & topological graph matching (labeled 1 to n with circular points over photos, NO overlay).
4. Storing crater similarity description probabilities.
5. End-to-end multi-image pipeline execution.
"""
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from scipy.spatial import Delaunay

from app.config import PipelineConfig
from app.pipeline.pds4_reader import PDS4Reader, LunarImageProduct
from app.pipeline.preprocessor import ImagePreprocessor
from app.pipeline.pairing import AnchorPairingManager, LunarImageItem, ImagePair
from app.pipeline.superpoint import SuperPointPipeline
from app.pipeline.loftr import LoFTRPipeline
from app.pipeline.geometric import GeometricVerifier, GeometricVerificationResult
from app.pipeline.evaluation import RegistrationEvaluator, AlgorithmEvaluationReport
from app.pipeline.selector import AlgorithmSelector, SelectionDecision
from app.pipeline.registration import ImageRegistrar, RegistrationResult
from app.pipeline.overlap import OverlapAnalyzer, OverlapResult
from app.pipeline.crater_detector import (
    LunarCraterDetector,
    CraterDetectionResult,
    DetectedCrater,
    MatchedCraterFeature
)
from app.pipeline.crater_trainer import CraterTrainingManager

@dataclass
class SingleImageResult:
    """Result of processing an individual lunar image."""
    id: str
    name: str
    image: np.ndarray                 # Preprocessed float32 [0, 1]
    raw_image: np.ndarray
    metadata: Dict[str, Any]
    craters: List[DetectedCrater]
    craters_result: CraterDetectionResult
    keypoints: np.ndarray             # (K, 2) feature locations
    descriptors: np.ndarray           # (K, 256) feature descriptors
    keypoint_scores: np.ndarray       # (K,) feature response scores
    annotated_features: np.ndarray    # RGB image with detected circular feature points and crater rims
    execution_time_sec: float

@dataclass
class GraphNode:
    """A matched feature / crater point labeled from 1 to n across two photos."""
    label: int                        # 1 to n integer label
    pt_a: Tuple[float, float]         # (x, y) on Photo 1
    pt_b: Tuple[float, float]         # (x, y) on Photo 2
    is_crater: bool
    crater_a_id: Optional[str]
    crater_b_id: Optional[str]
    diameter_a: float
    diameter_b: float
    rim_contrast_a: float
    rim_contrast_b: float
    similarity_probability: float     # [0.0, 1.0] probability of representing the same crater
    similarity_description: str       # Morphometric & appearance similarity explanation

@dataclass
class TwoImageGraphMatchResult:
    """Result of matching two specific images with circular points labeled 1..n and topological graph."""
    image_a_id: str
    image_a_name: str
    image_b_id: str
    image_b_name: str
    total_matched_features: int
    matched_craters_count: int
    nodes: List[GraphNode]
    edges: List[Tuple[int, int]]       # 1-based (u, v) pairs connecting nodes
    photo_a_graph: np.ndarray          # Photo 1 with graph & labeled circular points
    photo_b_graph: np.ndarray          # Photo 2 with graph & labeled circular points
    side_by_side_graph: np.ndarray     # Photo 1 and Photo 2 side-by-side (NO overlay!)
    homography: Optional[np.ndarray]
    inlier_count: int
    execution_time_sec: float
    is_matched: bool = True            # False if images depict different surfaces / no geometric alignment
    status_message: str = ""           # Explanation of match or misalignment status

@dataclass
class PairPipelineResult:
    pair_id: str
    target_id: str
    target_name: str
    superpoint_matches: int
    loftr_matches: int
    superpoint_geom: GeometricVerificationResult
    loftr_geom: GeometricVerificationResult
    superpoint_eval: AlgorithmEvaluationReport
    loftr_eval: AlgorithmEvaluationReport
    decision: SelectionDecision
    registration: RegistrationResult
    overlap: OverlapResult
    crater_detection: CraterDetectionResult
    execution_time_sec: float

@dataclass
class FullPipelineJobResult:
    job_id: str
    anchor_item: LunarImageItem
    processed_items: List[LunarImageItem]
    pair_results: List[PairPipelineResult]
    total_execution_time_sec: float
    status: str

class LunarPipelineManager:
    """Master controller managing lunar image processing, crater detection, and two-image graph matching."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.superpoint = SuperPointPipeline(self.config.superpoint)
        self.loftr = LoFTRPipeline(self.config.loftr)
        self.verifier = GeometricVerifier(self.config.ransac)
        self.evaluator = RegistrationEvaluator(self.config.evaluation)
        self.crater_detector = LunarCraterDetector(self.config.crater_detector)
        self.trainer = CraterTrainingManager()
        # In-memory storage of processed images and cross-image similarity records
        self.cached_single_images: Dict[str, SingleImageResult] = {}
        self.cached_similarity_data: Dict[str, Any] = {}

    def process_single_image(self, product: LunarImageProduct, item_id: Optional[str] = None) -> SingleImageResult:
        """Process one image at a time: run calibration, parallel native NLM, percentile norm, crater detection, and feature extraction."""
        t_start = time.time()
        item_id = item_id or f"img_{product.name}"

        # Step 1: Preprocess (Calibrate -> Native Parallel NLM -> Percentile Norm, no Hapke)
        norm_img, prep_log = self.preprocessor.process(product)

        # Step 2: Independent Crater Detection on this image
        crater_res = self.crater_detector.detect(norm_img)

        # Step 2b: Enrich craters with deep angle-invariant neural embeddings
        ha, wa = norm_img.shape[:2]
        for c in crater_res.craters:
            cx, cy, r = int(round(c.center_x)), int(round(c.center_y)), max(4, int(round(c.radius)))
            pad = int(r * 1.3)
            x0, x1 = max(0, cx - pad), min(wa, cx + pad + 1)
            y0, y1 = max(0, cy - pad), min(ha, cy + pad + 1)
            roi = norm_img[y0:y1, x0:x1]
            if roi.shape[0] >= 4 and roi.shape[1] >= 4:
                patch = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
                c.descriptor = self.trainer.compute_embedding(patch)

        # Step 3: Independent Feature Extraction (SuperPoint keypoints & descriptors)
        kps, desc, scores = self.superpoint.extract_features(norm_img)

        # Step 4: Render clean annotated feature preview
        ha, wa = norm_img.shape[:2]
        u8 = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)
        annotated = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

        # Draw detected crater rims with circular points
        for c in crater_res.craters:
            cx, cy, r = int(round(c.center_x)), int(round(c.center_y)), max(4, int(round(c.radius)))
            # Circular outer rim
            cv2.circle(annotated, (cx, cy), r, (0, 230, 115), 2, lineType=cv2.LINE_AA)
            # Center circular point
            cv2.circle(annotated, (cx, cy), 3, (0, 165, 255), -1, lineType=cv2.LINE_AA)
            # Crater ID label
            cv2.putText(
                annotated,
                f"{c.crater_id} (D={c.diameter:.0f})",
                (cx - r, max(14, cy - r - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 230, 115),
                1,
                lineType=cv2.LINE_AA
            )

        # Draw prominent keypoints as clean circular dots
        for pt in kps[:60]:
            kx, ky = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(annotated, (kx, ky), 2, (255, 200, 0), -1, lineType=cv2.LINE_AA)

        exec_time = round(time.time() - t_start, 3)

        result = SingleImageResult(
            id=item_id,
            name=product.name,
            image=norm_img,
            raw_image=product.raw_image,
            metadata={
                "pds4": product.metadata.__dict__,
                "preprocessing": prep_log
            },
            craters=crater_res.craters,
            craters_result=crater_res,
            keypoints=kps,
            descriptors=desc,
            keypoint_scores=scores,
            annotated_features=annotated,
            execution_time_sec=exec_time
        )
        self.cached_single_images[item_id] = result
        return result

    def process_n_images(self, products: List[LunarImageProduct]) -> List[SingleImageResult]:
        """Process all n input lunar images one at a time and cache their individual detection results."""
        self.cached_single_images.clear()
        self.cached_similarity_data.clear()
        results = []
        for i, prod in enumerate(products):
            item_id = f"img_{i}_{prod.name}"
            res = self.process_single_image(prod, item_id=item_id)
            results.append(res)
        return results

    @staticmethod
    def _render_graph_on_photo(
        img: np.ndarray,
        nodes: List[GraphNode],
        edges: List[Tuple[int, int]],
        is_target: bool = False,
        title: str = ""
    ) -> np.ndarray:
        """Render topological graph over a lunar photo with circular points labeled 1 to n."""
        ha, wa = img.shape[:2]
        if img.dtype != np.uint8:
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            u8 = img.copy()

        if u8.ndim == 2:
            canvas = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        elif u8.ndim == 3 and u8.shape[2] == 4:
            canvas = cv2.cvtColor(u8, cv2.COLOR_BGRA2BGR)
        elif u8.ndim == 3 and u8.shape[2] == 1:
            canvas = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        else:
            canvas = u8[:, :, :3].copy()

        # Map label -> (x, y) coordinates
        node_map = {n.label: (n.pt_b if is_target else n.pt_a) for n in nodes}

        # 1. Draw Delaunay topological graph edges over photo
        for u_idx, v_idx in edges:
            if u_idx in node_map and v_idx in node_map:
                pt1 = (int(round(node_map[u_idx][0])), int(round(node_map[u_idx][1])))
                pt2 = (int(round(node_map[v_idx][0])), int(round(node_map[v_idx][1])))
                # Cyan/Gold graph edge line
                cv2.line(canvas, pt1, pt2, (255, 200, 0), 2, lineType=cv2.LINE_AA)

        # 2. Draw circular feature points and numbered badges (1 to n)
        for n in nodes:
            pt = node_map[n.label]
            cx, cy = int(round(pt[0])), int(round(pt[1]))

            # Outer circular halo for feature
            halo_color = (0, 230, 115) if n.is_crater else (0, 165, 255)
            cv2.circle(canvas, (cx, cy), 12, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 12, halo_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            # Circular label badge (1 to n)
            badge_r = 10
            badge_x = min(wa - badge_r - 2, max(badge_r + 2, cx + 15))
            badge_y = max(badge_r + 2, cy - 13)
            cv2.circle(canvas, (badge_x, badge_y), badge_r, (15, 20, 28), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (badge_x, badge_y), badge_r, halo_color, 1, lineType=cv2.LINE_AA)

            lbl_str = str(n.label)
            font_scale = 0.38 if len(lbl_str) <= 2 else 0.30
            text_size = cv2.getTextSize(lbl_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = badge_x - text_size[0] // 2
            ty = badge_y + text_size[1] // 2
            cv2.putText(canvas, lbl_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Title banner
        if title:
            cv2.rectangle(canvas, (0, 0), (wa, 26), (15, 18, 26), -1)
            cv2.putText(canvas, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, lineType=cv2.LINE_AA)

        return canvas

    def match_two_specific_images(
        self,
        img_res_a: SingleImageResult,
        img_res_b: SingleImageResult,
        max_graph_nodes: int = 24
    ) -> TwoImageGraphMatchResult:
        """Compute feature & crater correspondences between two specific images.
        
        Renders two photos side by side (NO overlay!), marks features with circular points,
        labels each circle from 1 to n, and displays the matching graph over each photo.
        Calculates and stores the similarity description probability of being the same crater.
        """
        t_start = time.time()
        img_a = img_res_a.image
        img_b = img_res_b.image

        # 1. Match features using SuperPoint and LoFTR
        sp_matches_a, sp_matches_b, sp_scores = self.superpoint.match(
            img_res_a.keypoints, img_res_a.descriptors,
            img_res_b.keypoints, img_res_b.descriptors
        )

        loftr_res = self.loftr.process_pair(img_a, img_b)
        loftr_matches_a = loftr_res["matched_pts_anchor"]
        loftr_matches_b = loftr_res["matched_pts_target"]

        # Merge candidate correspondences
        if len(sp_matches_a) > 0 and len(loftr_matches_a) > 0:
            cand_a = np.vstack([sp_matches_a, loftr_matches_a])
            cand_b = np.vstack([sp_matches_b, loftr_matches_b])
        elif len(sp_matches_a) > 0:
            cand_a = sp_matches_a
            cand_b = sp_matches_b
        elif len(loftr_matches_a) > 0:
            cand_a = loftr_matches_a
            cand_b = loftr_matches_b
        else:
            cand_a = np.empty((0, 2), dtype=np.float32)
            cand_b = np.empty((0, 2), dtype=np.float32)

        # 2. Geometric Verification with RANSAC
        geom = self.verifier.verify("Combined", cand_a, cand_b)
        eval_report = self.evaluator.evaluate(
            geom,
            target_shape=img_b.shape[:2],
            anchor_shape=img_a.shape[:2]
        )

        # Check if the images represent a valid geometric correspondence (same lunar surface)
        is_geom_valid = bool(
            geom.is_valid and
            eval_report.plausibility.is_plausible and
            geom.inlier_count >= self.config.ransac.min_inliers_required and
            geom.reprojection_rmse <= 4.5 and
            geom.inlier_ratio >= 0.10
        )

        H = geom.homography if is_geom_valid else None
        inlier_mask = geom.inlier_mask if is_geom_valid else np.zeros((len(cand_a),), dtype=bool)

        # 3. Crater-to-Crater Matching and Similarity Description Probability
        nodes: List[GraphNode] = []
        node_label = 1
        matched_crater_pairs = []

        craters_a = img_res_a.craters
        craters_b = img_res_b.craters
        matched_b_craters = set()

        # Pre-compute H_inv for mapping points from Anchor (A) -> Target (B)
        H_inv = None
        if H is not None:
            try:
                det = float(np.linalg.det(H))
                if abs(det) > 1e-7 and not np.isnan(det) and not np.isinf(det):
                    H_inv = np.linalg.inv(H)
            except Exception:
                H_inv = None

        # Craters can ONLY be matched if a valid spatial transformation H is verified
        if H_inv is not None:
            for ca in craters_a:
                best_cb = None
                best_prob = 0.0
                best_desc = ""
                best_dist = float("inf")

                try:
                    pt_a_h = np.array([ca.center_x, ca.center_y, 1.0], dtype=np.float32)
                    pt_b_pred = H_inv @ pt_a_h
                    if abs(pt_b_pred[2]) > 1e-6 and not np.isnan(pt_b_pred[2]):
                        px = float(pt_b_pred[0] / pt_b_pred[2])
                        py = float(pt_b_pred[1] / pt_b_pred[2])
                        hb, wb = img_b.shape[:2]
                        if -20 <= px <= wb + 20 and -20 <= py <= hb + 20:
                            for idx_b, cb in enumerate(craters_b):
                                if idx_b in matched_b_craters:
                                    continue
                                dist = float(np.hypot(cb.center_x - px, cb.center_y - py))
                                prob, desc = LunarCraterDetector.compute_similarity(ca, cb, spatial_distance=dist)
                                if prob > best_prob and prob >= 0.50:
                                    best_prob = prob
                                    best_desc = desc
                                    best_cb = (idx_b, cb)
                                    best_dist = dist
                except Exception:
                    pass

                if best_cb is not None and best_prob >= 0.50:
                    idx_b, cb = best_cb
                    matched_b_craters.add(idx_b)
                    nodes.append(GraphNode(
                        label=node_label,
                        pt_a=(float(np.nan_to_num(ca.center_x)), float(np.nan_to_num(ca.center_y))),
                        pt_b=(float(np.nan_to_num(cb.center_x)), float(np.nan_to_num(cb.center_y))),
                        is_crater=True,
                        crater_a_id=ca.crater_id,
                        crater_b_id=cb.crater_id,
                        diameter_a=float(np.nan_to_num(ca.diameter)),
                        diameter_b=float(np.nan_to_num(cb.diameter)),
                        rim_contrast_a=float(np.nan_to_num(ca.rim_contrast)),
                        rim_contrast_b=float(np.nan_to_num(cb.rim_contrast)),
                        similarity_probability=round(float(np.nan_to_num(best_prob, nan=0.5)), 4),
                        similarity_description=best_desc
                    ))
                    matched_crater_pairs.append({
                        "label": node_label,
                        "crater_a": ca.crater_id,
                        "crater_b": cb.crater_id,
                        "similarity_probability": round(float(np.nan_to_num(best_prob, nan=0.5)), 4),
                        "description": best_desc
                    })
                    node_label += 1

        # 4. Complement with Top Geometrically Verified Feature Keypoints
        if is_geom_valid and len(geom.anchor_inliers) > 0:
            remaining_slots = max(0, max_graph_nodes - len(nodes))
            inliers_a = geom.anchor_inliers
            inliers_b = geom.target_inliers

            # Subsample inliers evenly across the space
            step = max(1, len(inliers_a) // max(1, remaining_slots))
            selected_indices = list(range(0, len(inliers_a), step))[:remaining_slots]

            for s_idx in selected_indices:
                pa = inliers_a[s_idx]
                pb = inliers_b[s_idx]

                # Reprojection residual: H maps Target (B) -> Anchor (A)
                res = 1.5
                if H is not None:
                    try:
                        p_pred = H @ np.array([pb[0], pb[1], 1.0], dtype=np.float32)
                        if abs(p_pred[2]) > 1e-6:
                            px = float(p_pred[0] / p_pred[2])
                            py = float(p_pred[1] / p_pred[2])
                            res = float(np.hypot(pa[0] - px, pa[1] - py))
                    except Exception:
                        res = 1.5

                if np.isnan(res) or np.isinf(res) or res > 4.5:
                    continue

                feat_prob = float(np.clip(1.0 - res / 8.0, 0.60, 0.98))
                nodes.append(GraphNode(
                    label=node_label,
                    pt_a=(float(np.nan_to_num(pa[0])), float(np.nan_to_num(pa[1]))),
                    pt_b=(float(np.nan_to_num(pb[0])), float(np.nan_to_num(pb[1]))),
                    is_crater=False,
                    crater_a_id=None,
                    crater_b_id=None,
                    diameter_a=0.0,
                    diameter_b=0.0,
                    rim_contrast_a=0.0,
                    rim_contrast_b=0.0,
                    similarity_probability=round(feat_prob, 4),
                    similarity_description=f"Geometric inlier feature match (P={feat_prob*100:.1f}%, reprojection residual={res:.2f}px)"
                ))
                node_label += 1

        # Store similarity description probability data in memory
        pair_key = f"{img_res_a.name}_vs_{img_res_b.name}"
        self.cached_similarity_data[pair_key] = {
            "matched_craters": matched_crater_pairs,
            "total_nodes": len(nodes),
            "execution_time_sec": round(time.time() - t_start, 3)
        }

        # 5. Construct Topological Graph (Delaunay Triangulation over nodes)
        pts_a = np.array([n.pt_a for n in nodes], dtype=np.float32) if len(nodes) > 0 else np.empty((0, 2), dtype=np.float32)
        edges: List[Tuple[int, int]] = []
        ha, wa = img_a.shape[:2]

        if len(pts_a) >= 3:
            try:
                tri = Delaunay(pts_a)
                edge_set = set()
                max_edge_len = 0.55 * max(ha, wa)
                for simplex in tri.simplices:
                    for i in range(3):
                        u = int(simplex[i])
                        v = int(simplex[(i + 1) % 3])
                        if u > v:
                            u, v = v, u
                        p1, p2 = pts_a[u], pts_a[v]
                        dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                        if dist < max_edge_len:
                            edge_set.add((u + 1, v + 1))  # 1-based labels
                edges = sorted(list(edge_set))
            except Exception:
                # Fallback to chain edges if Delaunay fails (e.g. collinear points)
                edges = [(k, k + 1) for k in range(1, len(nodes))]
        elif len(pts_a) == 2:
            edges = [(1, 2)]

        # 6. Render Graph on Photo 1 and Photo 2 (NO OVERLAY!)
        title_a = f"Image 1: {img_res_a.name} ({len(nodes)} Features Labeled 1..{len(nodes)})" if len(nodes) > 0 else f"Image 1: {img_res_a.name} (0 Common Features)"
        title_b = f"Image 2: {img_res_b.name} ({len(nodes)} Features Labeled 1..{len(nodes)})" if len(nodes) > 0 else f"Image 2: {img_res_b.name} (0 Common Features)"

        photo_a_graph = self._render_graph_on_photo(
            img_a, nodes, edges, is_target=False,
            title=title_a
        )
        photo_b_graph = self._render_graph_on_photo(
            img_b, nodes, edges, is_target=True,
            title=title_b
        )

        # 7. Render Side-by-Side Dual Photo Match (NO OVERLAY)
        hb, wb = photo_b_graph.shape[:2]
        h_max = max(ha, hb)
        sep_w = 6
        side_by_side = np.zeros((h_max, wa + wb + sep_w, 3), dtype=np.uint8)
        side_by_side[:ha, :wa, :] = photo_a_graph[:, :, :3]
        side_by_side[:, wa : wa + sep_w, :] = (50, 60, 80)  # Separator line
        side_by_side[:hb, wa + sep_w : wa + sep_w + wb, :] = photo_b_graph[:, :, :3]

        if len(nodes) == 0:
            banner_h = 32
            cv2.rectangle(side_by_side, (0, h_max - banner_h), (wa + wb + sep_w, h_max), (15, 18, 30), -1)
            msg = "DIFFERENT LUNAR SURFACES / NO COMMON OVERLAP (0 MATCHES)"
            t_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            tx = max(10, (wa + wb + sep_w - t_size[0]) // 2)
            cv2.putText(side_by_side, msg, (tx, h_max - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, lineType=cv2.LINE_AA)

        exec_time = round(time.time() - t_start, 3)

        return TwoImageGraphMatchResult(
            image_a_id=img_res_a.id,
            image_a_name=img_res_a.name,
            image_b_id=img_res_b.id,
            image_b_name=img_res_b.name,
            total_matched_features=len(nodes),
            matched_craters_count=len(matched_crater_pairs),
            nodes=nodes,
            edges=edges,
            photo_a_graph=photo_a_graph,
            photo_b_graph=photo_b_graph,
            side_by_side_graph=side_by_side,
            homography=H,
            inlier_count=geom.inlier_count if is_geom_valid else 0,
            execution_time_sec=exec_time,
            is_matched=len(nodes) > 0,
            status_message="Valid surface correspondence established" if len(nodes) > 0 else "No common surface features detected between frames (different lunar surfaces or no overlap)"
        )

    def process_products(
        self,
        products: List[LunarImageProduct],
        anchor_identifier: Optional[Any] = 0,
        job_id: Optional[str] = None
    ) -> FullPipelineJobResult:
        """Execute the full pipeline across all input lunar image products."""
        start_time = time.time()
        job_id = job_id or f"job_{int(start_time)}"

        # Clear cached images from any previous run so current batch is completely isolated
        self.cached_single_images.clear()
        self.cached_similarity_data.clear()

        # Step 2: Preprocess all input images one at a time & cache features
        processed_items: List[LunarImageItem] = []
        for i, prod in enumerate(products):
            item_id = f"img_{i}_{prod.name}"
            s_res = self.process_single_image(prod, item_id=item_id)
            item = LunarImageItem(
                id=item_id,
                name=prod.name,
                image=s_res.image,
                raw_image=s_res.raw_image,
                metadata=s_res.metadata
            )
            processed_items.append(item)

        # Step 3: Anchor-Based Image Pairing
        anchor_item, pairs = AnchorPairingManager.create_pairs(processed_items, anchor_identifier)

        pair_results: List[PairPipelineResult] = []

        # Process each pair against the designated anchor
        for pair in pairs:
            p_start = time.time()
            img_a = pair.anchor.image
            img_b = pair.target.image

            # Step 4: Run TWO Independent Correspondence Methods IN PARALLEL
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_sp = executor.submit(self.superpoint.process_pair, img_a, img_b)
                future_loftr = executor.submit(self.loftr.process_pair, img_a, img_b)

                sp_res = future_sp.result()
                loftr_res = future_loftr.result()

            # Step 5: Geometric Verification (RANSAC independently on each)
            geom_sp = self.verifier.verify(
                "SuperPoint",
                sp_res["matched_pts_anchor"],
                sp_res["matched_pts_target"]
            )
            geom_loftr = self.verifier.verify(
                "LoFTR",
                loftr_res["matched_pts_anchor"],
                loftr_res["matched_pts_target"]
            )

            # Step 6: Multi-Metric Evaluation of SuperPoint and LoFTR
            eval_sp = self.evaluator.evaluate(
                geom_sp,
                target_shape=img_b.shape,
                anchor_shape=img_a.shape
            )
            eval_loftr = self.evaluator.evaluate(
                geom_loftr,
                target_shape=img_b.shape,
                anchor_shape=img_a.shape
            )

            # Step 7: Algorithm Selection (Select the most reliable result)
            decision = AlgorithmSelector.select_best(
                geom_sp, eval_sp,
                geom_loftr, eval_loftr
            )

            # Step 8: Image Registration
            chosen_H = decision.selected_homography
            is_valid_pair = bool(
                chosen_H is not None and
                decision.selected_geom_result.is_valid and
                decision.selected_evaluation.plausibility.is_plausible and
                decision.confidence_level != "LOW"
            )

            if is_valid_pair:
                reg_result = ImageRegistrar.warp_target_to_anchor(
                    img_b,
                    chosen_H,
                    anchor_shape=img_a.shape
                )
                overlap_result = OverlapAnalyzer.compute_overlap(
                    img_a,
                    reg_result.warped_target,
                    reg_result.warped_mask
                )
                crater_result = self.crater_detector.detect(
                    anchor_img=img_a,
                    warped_target=reg_result.warped_target,
                    overlap_mask=overlap_result.overlap_mask
                )
            else:
                # Pair depicts different surfaces or has no geometric alignment
                ha, wa = img_a.shape[:2]
                reg_result = RegistrationResult(
                    warped_target=np.zeros_like(img_a),
                    warped_mask=np.zeros((ha, wa), dtype=np.uint8),
                    anchor_shape=img_a.shape,
                    homography=np.eye(3, dtype=np.float32)
                )
                overlap_result = OverlapResult(
                    overlap_mask=np.zeros((ha, wa), dtype=np.uint8),
                    overlap_area_pixels=0,
                    overlap_ratio_anchor=0.0,
                    overlap_ratio_target=0.0,
                    bounding_box=(0, 0, 0, 0),
                    contour_points=[],
                    has_valid_overlap=False,
                    blended_preview=img_a
                )
                crater_result = self.crater_detector.detect(
                    anchor_img=img_a,
                    warped_target=None,
                    overlap_mask=None
                )

            p_end = time.time()
            pair_results.append(PairPipelineResult(
                pair_id=pair.pair_id,
                target_id=pair.target.id,
                target_name=pair.target.name,
                superpoint_matches=sp_res["num_candidates"],
                loftr_matches=loftr_res["num_candidates"],
                superpoint_geom=geom_sp,
                loftr_geom=geom_loftr,
                superpoint_eval=eval_sp,
                loftr_eval=eval_loftr,
                decision=decision,
                registration=reg_result,
                overlap=overlap_result,
                crater_detection=crater_result,
                execution_time_sec=round(p_end - p_start, 3)
            ))

        total_time = round(time.time() - start_time, 3)
        return FullPipelineJobResult(
            job_id=job_id,
            anchor_item=anchor_item,
            processed_items=processed_items,
            pair_results=pair_results,
            total_execution_time_sec=total_time,
            status="SUCCESS"
        )
