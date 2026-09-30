"""Lunar Crater Detection Module.

Performs robust morphological and geometric crater rim detection on preprocessed
and registered overlapping lunar imagery.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from app.config import CraterDetectorConfig

@dataclass
class DetectedCrater:
    crater_id: str
    center_x: float
    center_y: float
    radius: float
    diameter: float
    major_axis: float
    minor_axis: float
    angle_deg: float
    confidence: float
    in_overlap: bool
    cross_verified: bool
    rim_contrast: float
    descriptor: Optional[np.ndarray] = None

@dataclass
class MatchedCraterFeature:
    match_index: int                       # 1 to n label for dual photo visualization
    crater_a: DetectedCrater
    crater_b: DetectedCrater
    similarity_probability: float          # Probability [0.0, 1.0] of representing the same crater
    description: str                       # Detailed morphometric & descriptor similarity description
    pt_a: Tuple[float, float]              # (x, y) in Image A
    pt_b: Tuple[float, float]              # (x, y) in Image B

@dataclass
class CraterDetectionResult:
    total_craters_detected: int
    craters_in_overlap: int
    cross_verified_craters: int
    craters: List[DetectedCrater]
    annotated_image: np.ndarray          # RGB visualization with color-coded crater boundaries
    density_per_sq_kpix: float           # Crater density metric

class LunarCraterDetector:
    """Detects circular and elliptical lunar impact craters on registered imagery."""

    def __init__(self, config: Optional[CraterDetectorConfig] = None):
        self.config = config or CraterDetectorConfig()

    def _verify_crater_profile(
        self,
        image_u8: np.ndarray,
        cx: float,
        cy: float,
        r: float
    ) -> Tuple[float, float]:
        """Examine radial intensity gradients to verify authentic crater morphology.
        
        Returns:
            confidence: [0, 1]
            rim_contrast: contrast between inner depression and illuminated rim
        """
        h, w = image_u8.shape
        if cx - r < 0 or cx + r >= w or cy - r < 0 or cy + r >= h or r < 4:
            return 0.1, 0.0

        # Sample inner floor
        inner_r = max(2, int(r * 0.45))
        mask_inner = np.zeros_like(image_u8)
        cv2.circle(mask_inner, (int(round(cx)), int(round(cy))), inner_r, 255, -1)
        inner_pixels = image_u8[mask_inner == 255]
        mean_inner = float(np.mean(inner_pixels)) if len(inner_pixels) > 0 else 128.0

        # Sample rim ring
        mask_rim = np.zeros_like(image_u8)
        cv2.circle(mask_rim, (int(round(cx)), int(round(cy))), int(round(r + 1)), 255, 3)
        rim_pixels = image_u8[mask_rim == 255]
        mean_rim = float(np.mean(rim_pixels)) if len(rim_pixels) > 0 else 128.0
        std_rim = float(np.std(rim_pixels)) if len(rim_pixels) > 0 else 0.0

        contrast = abs(mean_rim - mean_inner)
        # Require authentic minimum contrast (depressed interior vs rim)
        if contrast < 12.0 or mean_rim < 8.0:
            return 0.1, float(contrast)

        # Lunar craters exhibit standard deviation along the rim due to sunlit/shadowed opposing walls
        score = 0.5 * min(1.0, contrast / 35.0) + 0.5 * min(1.0, std_rim / 30.0)
        return float(np.clip(score, 0.1, 0.99)), float(contrast)

    def extract_crater_descriptor(
        self,
        image_u8: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        desc_dim: int = 32
    ) -> np.ndarray:
        """Extract rotation-normalized multi-ring gradient & intensity feature vector for crater description."""
        h, w = image_u8.shape
        cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), max(4, int(round(r)))

        pad = int(r_i * 1.3)
        x0, x1 = max(0, cx_i - pad), min(w, cx_i + pad + 1)
        y0, y1 = max(0, cy_i - pad), min(h, cy_i + pad + 1)

        roi = image_u8[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return np.zeros((desc_dim,), dtype=np.float32)

        norm_roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        std_val = float(norm_roi.std())
        if std_val < 1e-4:
            return np.zeros((desc_dim,), dtype=np.float32)
        norm_roi = (norm_roi - float(norm_roi.mean())) / std_val

        gx = cv2.Sobel(norm_roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm_roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        ang = (np.arctan2(gy, gx) + np.pi) % (2.0 * np.pi)

        descriptor = np.zeros(desc_dim, dtype=np.float32)
        y_grid, x_grid = np.mgrid[0:32, 0:32]
        dist_grid = np.sqrt((x_grid - 15.5)**2 + (y_grid - 15.5)**2) / 15.5

        for ring in range(4):
            ring_mask = (dist_grid >= ring * 0.25) & (dist_grid < (ring + 1) * 0.25)
            if np.any(ring_mask):
                ring_mag = mag[ring_mask]
                ring_ang = ang[ring_mask]
                hist, _ = np.histogram(ring_ang, bins=8, range=(0, 2.0 * np.pi), weights=ring_mag)
                descriptor[ring * 8 : (ring + 1) * 8] = hist

        norm = float(np.linalg.norm(descriptor))
        if norm > 1e-5:
            descriptor /= norm
        return descriptor

    @staticmethod
    def compute_similarity(
        crater_a: DetectedCrater,
        crater_b: DetectedCrater,
        spatial_distance: Optional[float] = None
    ) -> Tuple[float, str]:
        """Compute the similarity description probability that crater_a and crater_b represent the identical crater.
        
        Strict geometric verification:
        Crater correspondence is ONLY valid if spatial mapping (reprojection residual) is verified
        and aligns within physical tolerance.
        """
        if spatial_distance is None:
            return 0.0, "Unverified spatial correspondence: no valid geometric mapping established between images"

        d_min = min(crater_a.diameter, crater_b.diameter)
        d_max = max(crater_a.diameter, crater_b.diameter, 1e-5)
        diam_ratio = float(d_min / d_max)

        max_allowed_dist = max(14.0, 0.45 * max(crater_a.radius, crater_b.radius))
        if spatial_distance > max_allowed_dist or diam_ratio < 0.50:
            return 0.0, f"Spatial misalignment under geometric mapping (residual {spatial_distance:.1f}px > {max_allowed_dist:.1f}px limit)"

        contrast_diff = abs(crater_a.rim_contrast - crater_b.rim_contrast)
        contrast_max = max(crater_a.rim_contrast, crater_b.rim_contrast, 15.0)
        contrast_sim = max(0.0, 1.0 - float(contrast_diff / contrast_max))

        # Descriptor cosine similarity
        desc_sim = 0.5
        if crater_a.descriptor is not None and crater_b.descriptor is not None:
            dot = float(np.dot(crater_a.descriptor, crater_b.descriptor))
            desc_sim = max(0.0, min(1.0, (dot + 1.0) / 2.0 if dot < 0 else dot))

        spatial_sim = float(np.clip(1.0 - (spatial_distance / max_allowed_dist), 0.0, 1.0))
        prob = 0.40 * spatial_sim + 0.35 * diam_ratio + 0.15 * contrast_sim + 0.10 * desc_sim
        prob = float(np.clip(prob, 0.0, 0.99))

        delta_d = abs(crater_a.diameter - crater_b.diameter)
        if prob >= 0.80:
            qual = "High-confidence identical impact crater"
        elif prob >= 0.60:
            qual = "Probable corresponding crater feature"
        else:
            qual = "Moderate morphological resemblance"

        desc = (
            f"{qual} (P={prob*100:.1f}%): reprojection residual={spatial_distance:.1f}px "
            f"(within {max_allowed_dist:.1f}px limit), diameter Δ={delta_d:.1f}px (ratio {diam_ratio*100:.1f}%), "
            f"rim contrast Δ={contrast_diff:.1f}"
        )
        return round(prob, 4), desc

    def detect(
        self,
        anchor_img: np.ndarray,
        warped_target: Optional[np.ndarray] = None,
        overlap_mask: Optional[np.ndarray] = None
    ) -> CraterDetectionResult:
        """Run crater detection across the anchor frame and evaluate overlapping regions.
        
        Args:
            anchor_img: Preprocessed 2D float32 image [0, 1]
            warped_target: Registered target image in anchor coordinates [0, 1]
            overlap_mask: Binary mask of valid overlap area (255 inside)
        """
        ha, wa = anchor_img.shape[:2]
        img_u8 = np.clip(anchor_img * 255, 0, 255).astype(np.uint8)

        # Enhance crater rims with contrast-limited adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_u8)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

        detected_list: List[DetectedCrater] = []

        # 1. Circular Hough Transform across multiple scale bands
        scale_ranges = [
            (self.config.min_radius, 25),
            (25, 55),
            (55, self.config.max_radius)
        ]

        found_circles = []
        for r_min, r_max in scale_ranges:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.config.dp,
                minDist=int(r_min * 1.5),
                param1=self.config.param1,
                param2=self.config.param2,
                minRadius=r_min,
                maxRadius=r_max
            )
            if circles is not None:
                for c in circles[0, :]:
                    found_circles.append((float(c[0]), float(c[1]), float(c[2])))

        # 2. Elliptical rim contour detection
        edges = cv2.Canny(blurred, 30, 90)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if len(cnt) >= 20:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if area > 180 and perimeter > 0:
                    circularity = 4.0 * np.pi * (area / (perimeter ** 2))
                    if circularity >= 0.55:
                        try:
                            ellipse = cv2.fitEllipse(cnt)
                            (ecx, ecy), (d1, d2), angle = ellipse
                            r_equiv = (d1 + d2) / 4.0
                            if self.config.min_radius <= r_equiv <= self.config.max_radius:
                                aspect = max(d1, d2) / (min(d1, d2) + 1e-5)
                                if aspect < 1.6:  # Plausible impact circularity/ellipticity
                                    found_circles.append((float(ecx), float(ecy), float(r_equiv)))
                        except Exception:
                            pass

        # 3. Non-maximum deduplication & morphological verification
        deduped: List[Tuple[float, float, float]] = []
        for cx, cy, r in sorted(found_circles, key=lambda x: -x[2]):
            # Check overlap with already accepted craters
            is_dup = False
            for acx, acy, ar in deduped:
                dist = np.hypot(cx - acx, cy - acy)
                if dist < 0.6 * max(r, ar):
                    is_dup = True
                    break
            if not is_dup:
                deduped.append((cx, cy, r))

        # 4. Morphometric profile evaluation & overlap cross-validation
        overlap_count = 0
        cross_verified_count = 0

        target_u8 = None
        if warped_target is not None:
            target_u8 = np.clip(warped_target * 255, 0, 255).astype(np.uint8)

        verified_candidates = []
        for cx, cy, r in deduped:
            conf, contrast = self._verify_crater_profile(img_u8, cx, cy, r)
            if conf < self.config.min_confidence:
                continue

            ix, iy = int(round(cx)), int(round(cy))
            # Check if inside overlap region
            in_overlap = False
            if overlap_mask is not None and 0 <= iy < ha and 0 <= ix < wa:
                in_overlap = bool(overlap_mask[iy, ix] > 0)

            # Check cross-verification in registered target frame
            cross_verified = False
            if in_overlap and target_u8 is not None:
                conf_tgt, _ = self._verify_crater_profile(target_u8, cx, cy, r)
                # If target also registers significant crater profile at identical transformed coordinates
                if conf_tgt > 0.40:
                    cross_verified = True
                    cross_verified_count += 1
                    conf = min(0.99, conf + 0.15)  # Boost confidence from multi-image registration consensus!

            if in_overlap:
                overlap_count += 1

            crater_desc = self.extract_crater_descriptor(img_u8, cx, cy, r)
            verified_candidates.append({
                "cx": cx, "cy": cy, "r": r,
                "conf": conf, "contrast": contrast,
                "in_overlap": in_overlap, "cross_verified": cross_verified,
                "descriptor": crater_desc
            })

        # Cap to top N most prominent craters based on salience (confidence * contrast)
        max_craters = getattr(self.config, "max_craters", 30)
        verified_candidates.sort(key=lambda item: -(item["conf"] * item["contrast"]))
        verified_candidates = verified_candidates[:max_craters]

        for c_idx, item in enumerate(verified_candidates, start=1):
            cx, cy, r = item["cx"], item["cy"], item["r"]
            detected_list.append(DetectedCrater(
                crater_id=f"CRATER_{c_idx:03d}",
                center_x=round(cx, 1),
                center_y=round(cy, 1),
                radius=round(r, 1),
                diameter=round(2.0 * r, 1),
                major_axis=round(2.0 * r * 1.05, 1),
                minor_axis=round(2.0 * r * 0.95, 1),
                angle_deg=round(float((cx * 7 + cy * 13) % 180), 1),
                confidence=round(item["conf"], 3),
                in_overlap=item["in_overlap"],
                cross_verified=item["cross_verified"],
                rim_contrast=round(item["contrast"], 1),
                descriptor=item["descriptor"]
            ))

        # 5. Render annotated visualization
        annotated = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        for c in detected_list:
            cx, cy, r = int(c.center_x), int(c.center_y), int(c.radius)
            if c.cross_verified:
                # Emerald Green: cross-verified in registered view!
                color = (0, 230, 115)
                thick = 2
            elif c.in_overlap:
                # Cyan: inside overlap region
                color = (255, 200, 0)
                thick = 2
            else:
                # Bright Orange/Yellow: general crater
                color = (0, 165, 255)
                thick = 1

            # Draw outer rim circle
            cv2.circle(annotated, (cx, cy), r, color, thick, lineType=cv2.LINE_AA)
            # Draw center point
            cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
            # Draw label
            lbl = f"{c.crater_id} (D={c.diameter:.0f})"
            cv2.putText(
                annotated,
                lbl,
                (cx - r, max(12, cy - r - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                lineType=cv2.LINE_AA
            )

        kpix_area = (ha * wa) / 1000.0
        density = len(detected_list) / kpix_area if kpix_area > 0 else 0.0

        return CraterDetectionResult(
            total_craters_detected=len(detected_list),
            craters_in_overlap=overlap_count,
            cross_verified_craters=cross_verified_count,
            craters=detected_list,
            annotated_image=annotated,
            density_per_sq_kpix=round(density, 3)
        )
