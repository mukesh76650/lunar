"""Image Preprocessor Module.

Implements the mandatory 4-step preprocessing sequence:
1. Radiometric Calibration
2. Hapke Photometric Correction (illumination and viewing-geometry normalization)
3. Non-Local Means (NLM) Denoising (edge-preserving crater texture filtering)
4. Percentile Normalization (1%–99% contrast stretching)
"""
import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple
from app.config import PreprocessingConfig
from app.pipeline.pds4_reader import LunarImageProduct, PDS4Metadata

class ImagePreprocessor:
    """Preprocesses raw lunar images for robust feature matching and crater detection."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        # Enable OpenCV internal parallel optimization for high-performance denoising
        if getattr(self.config, "use_opencv_parallel", True):
            cv2.setUseOptimized(True)
            threads = getattr(self.config, "opencv_threads", 0)
            if threads <= 0:
                threads = cv2.getNumberOfCPUs()
            cv2.setNumThreads(threads)

    def calibrate(self, image: np.ndarray, meta: Optional[PDS4Metadata] = None) -> np.ndarray:
        """Apply radiometric calibration using scale and offset factors."""
        if not self.config.apply_calibration:
            return image.copy()
            
        scale = self.config.calibration_scale
        offset = self.config.calibration_offset
        if meta is not None:
            if meta.scaling_factor != 1.0:
                scale = meta.scaling_factor
            if meta.scaling_offset != 0.0:
                offset = meta.scaling_offset
                
        calibrated = image * scale + offset
        return np.nan_to_num(calibrated, nan=0.0, posinf=1.0, neginf=0.0)

    def hapke_photometric_correction(
        self,
        image: np.ndarray,
        incidence_deg: Optional[float] = None,
        emission_deg: Optional[float] = None,
        phase_deg: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Deprecated: Hapke / Lommel-Seeliger photometric correction (removed from preprocessing sequence)."""
        return image.copy(), {"applied": False, "note": "Hapke correction model removed"}

    def nlm_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means (NLM) denoising at full native resolution with OpenCV parallel optimization."""
        if not self.config.apply_nlm:
            return image.copy()

        # Ensure OpenCV parallel computation optimization is enabled
        if getattr(self.config, "use_opencv_parallel", True):
            cv2.setUseOptimized(True)

        # Convert float image [0, 1] to uint8 for high-performance OpenCV fastNlMeans
        vmin, vmax = image.min(), image.max()
        if vmax - vmin > 1e-7:
            img_u8 = np.clip((image - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(image, dtype=np.uint8)

        # Native resolution denoising: fixed scaling has been completely removed
        # to prevent spatial downsampling and preserve sharp crater rims and subtle textures.
        denoised_u8 = cv2.fastNlMeansDenoising(
            img_u8,
            None,
            h=self.config.nlm_h,
            templateWindowSize=self.config.nlm_template_window_size,
            searchWindowSize=self.config.nlm_search_window_size
        )

        # Re-scale back to original float range
        denoised = denoised_u8.astype(np.float32) / 255.0 * (vmax - vmin) + vmin
        return denoised

    def percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply 1%–99% percentile contrast normalization scaling to [0, 1]."""
        if not self.config.apply_percentile_norm:
            vmin, vmax = image.min(), image.max()
            if vmax - vmin > 1e-7:
                return (image - vmin) / (vmax - vmin)
            return image

        p_low = np.percentile(image, self.config.percentile_low)
        p_high = np.percentile(image, self.config.percentile_high)

        if p_high - p_low > 1e-7:
            norm = np.clip((image - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            norm = np.zeros_like(image)
        return norm

    def process(self, product: LunarImageProduct) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the streamlined preprocessing sequence on a LunarImageProduct:
        1. Radiometric Calibration
        2. Non-Local Means (NLM) Denoising (at native resolution with OpenCV parallel optimization)
        3. Percentile Normalization (1–99% contrast stretching)
        
        Returns:
            processed_image: 2D float32 array in range [0, 1]
            metadata_log: Step-by-step processing parameters and execution log
        """
        raw = product.image
        meta = product.metadata

        # Step 1: Radiometric Calibration
        calibrated = self.calibrate(raw, meta)

        # Step 2: Non-Local Means (NLM) Denoising (Hapke photometric correction removed)
        denoised = self.nlm_denoise(calibrated)

        # Step 3: Percentile Normalization (1–99%)
        normalized = self.percentile_normalize(denoised)

        log = {
            "name": product.name,
            "dimensions": {"height": int(normalized.shape[0]), "width": int(normalized.shape[1])},
            "calibration": {"applied": self.config.apply_calibration},
            "hapke": {"applied": False, "status": "Removed from preprocessing"},
            "nlm_denoise": {
                "applied": self.config.apply_nlm,
                "h": self.config.nlm_h,
                "template_window": self.config.nlm_template_window_size,
                "search_window": self.config.nlm_search_window_size,
                "native_resolution": True,
                "opencv_parallel": cv2.useOptimized(),
                "opencv_threads": cv2.getNumThreads()
            },
            "percentile_norm": {
                "applied": self.config.apply_percentile_norm,
                "low": self.config.percentile_low,
                "high": self.config.percentile_high
            }
        }
        return normalized.astype(np.float32), log
